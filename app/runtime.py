from functools import lru_cache

from app.config import settings
from app.parsing import DocumentParser
from app.retrieval import LightRAGStore
from app.workflow import QAWorkflow


@lru_cache(maxsize=1)
def get_store() -> LightRAGStore:
    return LightRAGStore(
        data_dir=settings.data_dir,
        working_dir=settings.lightrag_working_dir,
        embedding_model=settings.stella_model_name,
        revision=settings.stella_revision,
        query_prompt_name=settings.stella_query_prompt_name,
        device=settings.stella_device,
        batch_size=settings.stella_batch_size,
        llm_timeout=settings.lightrag_llm_timeout,
        llm_max_async=settings.lightrag_llm_max_async,
        embedding_timeout=settings.lightrag_embedding_timeout,
        embedding_max_async=settings.lightrag_embedding_max_async,
        embedding_batch_num=settings.lightrag_embedding_batch_num,
        hf_token=settings.huggingface_token,
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
        query_mode=settings.lightrag_query_mode,
    )


@lru_cache(maxsize=1)
def get_parser() -> DocumentParser:
    return DocumentParser(
        raw_data_dir=settings.raw_data_dir,
        output_dir=settings.data_dir,
        assets_dir=settings.parsed_assets_dir,
        enable_image_ocr=settings.enable_image_ocr,
        enable_page_ocr_fallback=settings.enable_page_ocr_fallback,
    )


@lru_cache(maxsize=1)
def get_workflow() -> QAWorkflow:
    return QAWorkflow(
        store=get_store(),
        ollama_base_url=settings.ollama_base_url,
        ollama_model=settings.ollama_model,
    )
