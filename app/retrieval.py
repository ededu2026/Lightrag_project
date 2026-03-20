from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from threading import Lock
from typing import Any

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import wrap_embedding_func_with_attrs
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from app.chunking import markdown_chunking_func


@dataclass(slots=True)
class IngestProgress:
    running: bool = False
    stage: str = "idle"
    total_documents: int = 0
    completed_documents: int = 0
    current_item: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["progress"] = (
            0.0 if self.total_documents == 0 else self.completed_documents / self.total_documents
        )
        return payload


class LightRAGStore:
    def __init__(
        self,
        data_dir: Path,
        working_dir: Path,
        embedding_model: str,
        revision: str | None,
        query_prompt_name: str,
        device: str,
        batch_size: int,
        llm_timeout: int,
        llm_max_async: int,
        embedding_timeout: int,
        embedding_max_async: int,
        embedding_batch_num: int,
        hf_token: str | None,
        ollama_base_url: str,
        ollama_model: str,
        query_mode: str,
    ) -> None:
        self.data_dir = data_dir
        self.working_dir = working_dir
        self.embedding_model = embedding_model
        self.revision = revision
        self.query_prompt_name = query_prompt_name
        self.device = device
        self.resolved_device = self._resolve_device(device)
        self.batch_size = batch_size
        self.llm_timeout = llm_timeout
        self.llm_max_async = llm_max_async
        self.embedding_timeout = embedding_timeout
        self.embedding_max_async = embedding_max_async
        self.embedding_batch_num = embedding_batch_num
        self.hf_token = hf_token
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.query_mode = query_mode
        self.embeddings: SentenceTransformer | None = None
        self.rag: LightRAG | None = None
        self.ready = False
        self._lock = Lock()
        self._progress = IngestProgress()
        if hf_token:
            os.environ.setdefault("HF_TOKEN", hf_token)
        self.working_dir.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        if self.rag is None:
            self.rag = self._create_rag()
        await self.rag.initialize_storages()
        self.ready = self._graphml_path().exists()

    async def finalize(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()

    async def ingest(self) -> dict[str, int]:
        files = sorted(self.data_dir.glob("*.md"))
        if not files:
            self.ready = False
            self._set_progress(running=False, stage="idle", total_documents=0, completed_documents=0, current_item="", error="")
            return {"documents": 0}

        await self._reset_rag()
        self._set_progress(
            running=True,
            stage="ingesting",
            total_documents=len(files),
            completed_documents=0,
            current_item="",
            error="",
        )
        progress_bar = tqdm(total=len(files), desc="Ingesting markdown files", unit="doc", leave=True)
        try:
            for index, path in enumerate(files, start=1):
                self._set_progress(current_item=path.name, completed_documents=index - 1)
                content = path.read_text(encoding="utf-8")
                await self.rag.ainsert(content, ids=path.stem, file_paths=str(path))
                self._set_progress(completed_documents=index, current_item=path.name)
                progress_bar.set_postfix(file=path.name)
                progress_bar.update(1)
        except Exception as exc:
            self.ready = False
            progress_bar.close()
            self._set_progress(running=False, stage="failed", error=str(exc))
            raise
        progress_bar.close()
        self.ready = self._graphml_path().exists()
        self._set_progress(
            running=False,
            stage="completed",
            completed_documents=len(files),
            current_item="",
        )
        return {"documents": len(files)}

    async def query(self, question: str) -> dict[str, Any]:
        if self.rag is None:
            await self.initialize()
        if not self.ready:
            return {
                "answer": "Knowledge base is not ingested yet. Run Parse and Ingest first.",
                "contexts": [],
                "mode": self.query_mode,
            }
        param = QueryParam(mode=self.query_mode)
        answer = await self.rag.aquery(question, param=param)
        context = await self.rag.aquery_data(question, param=param)
        return {
            "answer": answer if isinstance(answer, str) else "",
            "contexts": self._normalize_contexts(context),
            "mode": self.query_mode,
        }

    def get_progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_dict()

    def export_graphml_path(self) -> Path:
        path = self._graphml_path()
        if not path.exists():
            raise FileNotFoundError("GraphML file is not available.")
        return path

    def export_json_path(self) -> Path:
        graphml_path = self.export_graphml_path()
        export_path = self.working_dir / "knowledge_graph.json"
        graph = nx.read_graphml(graphml_path)
        payload = nx.node_link_data(graph, edges="links")
        export_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return export_path

    async def _reset_rag(self) -> None:
        if self.rag is not None:
            await self.rag.finalize_storages()
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = self._create_rag()
        await self.rag.initialize_storages()
        self.ready = False

    def _create_rag(self) -> LightRAG:
        return LightRAG(
            working_dir=str(self.working_dir),
            graph_storage="NetworkXStorage",
            chunking_func=markdown_chunking_func,
            embedding_batch_num=self.embedding_batch_num,
            embedding_func_max_async=self.embedding_max_async,
            default_embedding_timeout=self.embedding_timeout,
            llm_model_max_async=self.llm_max_async,
            default_llm_timeout=self.llm_timeout,
            embedding_func=self._embedding_func(),
            llm_model_func=ollama_model_complete,
            llm_model_name=self.ollama_model,
            llm_model_kwargs={"host": self.ollama_base_url},
        )

    def _embedding_func(self):
        @wrap_embedding_func_with_attrs(
            embedding_dim=1024,
            max_token_size=8192,
            model_name=self.embedding_model,
        )
        async def embed(texts: list[str]) -> np.ndarray:
            model = self._get_embeddings()
            encoded = await asyncio.to_thread(
                model.encode,
                texts,
                batch_size=min(self.batch_size, len(texts)),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return np.array(encoded)

        return embed

    def _get_embeddings(self) -> SentenceTransformer:
        if self.embeddings is None:
            kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "device": self.resolved_device,
                "config_kwargs": {
                    "use_memory_efficient_attention": False,
                    "unpad_inputs": False,
                },
            }
            if self.revision:
                kwargs["revision"] = self.revision
            if self.hf_token:
                kwargs["token"] = self.hf_token
            self.embeddings = SentenceTransformer(self.embedding_model, **kwargs)
            self._repair_stella_position_ids(self.embeddings)
        return self.embeddings

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return device

    def _repair_stella_position_ids(self, model: SentenceTransformer) -> None:
        transformer = model._first_module()
        auto_model = transformer.auto_model
        embeddings = auto_model.embeddings
        size = getattr(auto_model.config, "max_position_embeddings", embeddings.position_ids.shape[0])
        embeddings.register_buffer(
            "position_ids",
            torch.arange(size, device=embeddings.position_ids.device),
            persistent=False,
        )

    def _graphml_path(self) -> Path:
        return self.working_dir / "graph_chunk_entity_relation.graphml"

    def _set_progress(self, **changes: Any) -> None:
        with self._lock:
            data = self._progress.to_dict()
            data.pop("progress", None)
            data.update(changes)
            self._progress = IngestProgress(**data)

    def _normalize_contexts(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        data = payload.get("data", {})
        chunks = data.get("chunks", [])
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        return [
            {
                "chunks": chunks,
                "entities": entities,
                "relationships": relationships,
                "references": data.get("references", []),
                "mode": payload.get("metadata", {}).get("query_mode", self.query_mode),
            }
        ]
