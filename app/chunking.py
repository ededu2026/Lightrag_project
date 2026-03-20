from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MD_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
RULE_RE = re.compile(r"^\s*---+\s*$")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    header_path: list[str]
    content: str
    kind: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class Document:
    doc_id: str
    source: str
    metadata: dict[str, Any]
    chunks: list[Chunk]


def _is_image_start(line: str) -> bool:
    stripped = line.strip()
    return bool(
        IMAGE_RE.search(stripped)
        or stripped.startswith("<img")
        or stripped.startswith("<img_summary>")
        or stripped.startswith("<image_summary>")
    )


def _collect_special_block(lines: list[str], index: int) -> tuple[list[str], int, str] | None:
    line = lines[index].rstrip("\n")
    stripped = line.strip()

    if stripped.startswith("<table>"):
        block = [line]
        index += 1
        while index < len(lines):
            block.append(lines[index].rstrip("\n"))
            if lines[index].strip().startswith("</table>"):
                return block, index + 1, "table"
            index += 1
        return block, index, "table"

    if stripped.startswith("<img_summary>"):
        block = [line]
        index += 1
        while index < len(lines):
            block.append(lines[index].rstrip("\n"))
            if lines[index].strip().startswith("</img_summary>"):
                return block, index + 1, "image"
            index += 1
        return block, index, "image"

    if stripped.startswith("<image_summary>"):
        block = [line]
        index += 1
        while index < len(lines):
            block.append(lines[index].rstrip("\n"))
            if lines[index].strip().startswith("</image_summary>"):
                return block, index + 1, "image"
            index += 1
        return block, index, "image"

    if _is_image_start(line):
        return [line], index + 1, "image"

    if MD_TABLE_RE.match(line):
        block = [line]
        index += 1
        while index < len(lines) and MD_TABLE_RE.match(lines[index].rstrip("\n")):
            block.append(lines[index].rstrip("\n"))
            index += 1
        return block, index, "table"

    return None


def _split_frontmatter(text: str) -> tuple[dict[str, Any], list[str]]:
    lines = text.splitlines()
    if len(lines) < 3 or lines[0].strip() != "---":
        return {}, lines

    closing = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if closing is None:
        return {}, lines

    raw = "\n".join(lines[1:closing])
    metadata = yaml.safe_load(raw) or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return metadata, lines[closing + 1 :]


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        normalized[key] = value
    return normalized


def chunk_markdown_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return chunk_markdown_text(text=text, source=str(path), doc_id=path.stem)


def chunk_markdown_text(text: str, source: str, doc_id: str) -> Document:
    metadata, lines = _split_frontmatter(text)
    header_path: list[str] = []
    chunks: list[Chunk] = []
    section_buffer: list[str] = []
    chunk_index = 0
    normalized_metadata = _normalize_metadata(metadata)

    def flush_section() -> None:
        nonlocal chunk_index
        content = "\n".join(section_buffer).strip()
        if not content:
            section_buffer.clear()
            return
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{chunk_index}",
                doc_id=doc_id,
                source=source,
                header_path=header_path.copy(),
                content=content,
                kind="text",
                metadata=normalized_metadata,
            )
        )
        chunk_index += 1
        section_buffer.clear()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if RULE_RE.match(line):
            flush_section()
            i += 1
            continue
        header = HEADER_RE.match(line)
        special = _collect_special_block(lines, i)

        if header:
            flush_section()
            level = len(header.group(1))
            title = header.group(2).strip()
            header_path[:] = header_path[: level - 1]
            header_path.append(title)
            i += 1
            continue

        if special:
            flush_section()
            block, i, kind = special
            content = "\n".join(block).strip()
            if content:
                chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}:{chunk_index}",
                            doc_id=doc_id,
                            source=source,
                            header_path=header_path.copy(),
                            content=content,
                            kind=kind,
                        metadata=normalized_metadata,
                    )
                )
                chunk_index += 1
            continue

        section_buffer.append(line)
        i += 1

    flush_section()
    return Document(
        doc_id=doc_id,
        source=source,
        metadata=normalized_metadata,
        chunks=chunks,
    )


def chunk_directory(data_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        documents.append(chunk_markdown_file(path))
    return documents


def markdown_chunking_func(
    tokenizer: Any,
    content: str,
    split_by_character: str | None,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict[str, Any]]:
    del tokenizer, split_by_character, split_by_character_only, chunk_overlap_token_size, chunk_token_size
    document = chunk_markdown_text(text=content, source="", doc_id="inline")
    chunks: list[dict[str, Any]] = []
    for index, chunk in enumerate(document.chunks):
        metadata_lines = [f"{key}: {value}" for key, value in chunk.metadata.items()]
        header = " > ".join(chunk.header_path) or "root"
        text = "\n".join(metadata_lines + [f"Header: {header}", f"Kind: {chunk.kind}", chunk.content]).strip()
        chunks.append(
            {
                "tokens": len(text.split()),
                "content": text,
                "chunk_order_index": index,
            }
        )
    return chunks
