from __future__ import annotations

from dataclasses import asdict, dataclass
from io import BytesIO
import re
from pathlib import Path
import sys
from threading import Lock
from typing import Any

import fitz
from PIL import Image
import pytesseract
from tqdm import tqdm
import yaml


HEADER_RE = re.compile(r"^\s*(\d+(\.\d+)*)?\s*[A-Z][A-Za-z0-9 ,:/()\-]{4,}$")
WORD_RE = re.compile(r"\S+")


@dataclass(slots=True)
class ParseResult:
    files: int
    markdowns: int
    images: int


@dataclass(slots=True)
class ParseProgress:
    running: bool = False
    stage: str = "idle"
    total_files: int = 0
    completed_files: int = 0
    current_file: str = ""
    markdowns: int = 0
    images: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["progress"] = 0.0 if self.total_files == 0 else self.completed_files / self.total_files
        return payload


class DocumentParser:
    def __init__(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        assets_dir: Path,
        enable_image_ocr: bool,
        enable_page_ocr_fallback: bool,
    ) -> None:
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        self.assets_dir = assets_dir
        self.enable_image_ocr = enable_image_ocr
        self.enable_page_ocr_fallback = enable_page_ocr_fallback
        self._lock = Lock()
        self._progress = ParseProgress()

    def parse_all(self) -> ParseResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assets_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            path
            for path in sorted(self.raw_data_dir.iterdir())
            if not path.name.startswith(".") and path.is_file()
        ]
        self._set_progress(
            running=True,
            stage="parsing",
            total_files=len(candidates),
            completed_files=0,
            current_file="",
            markdowns=0,
            images=0,
            error="",
        )

        files = 0
        markdowns = 0
        images = 0
        progress_bar = tqdm(
            total=len(candidates),
            desc="Parsing files",
            unit="file",
            leave=True,
            file=sys.stdout,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        try:
            for path in candidates:
                files += 1
                self._set_progress(current_file=path.name, completed_files=files - 1)
                if path.suffix.lower() == ".pdf":
                    image_count = self._parse_pdf(path)
                    markdowns += 1
                    images += image_count
                elif path.suffix.lower() == ".md":
                    target = self.output_dir / path.name
                    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
                    markdowns += 1
                self._set_progress(
                    completed_files=files,
                    markdowns=markdowns,
                    images=images,
                )
                progress_bar.set_postfix(file=path.name, markdowns=markdowns, images=images)
                progress_bar.update(1)
        except Exception as exc:
            progress_bar.close()
            self._set_progress(running=False, stage="failed", error=str(exc))
            raise
        progress_bar.close()

        self._set_progress(
            running=False,
            stage="completed",
            completed_files=files,
            current_file="",
            markdowns=markdowns,
            images=images,
        )
        return ParseResult(files=files, markdowns=markdowns, images=images)

    def get_progress(self) -> dict[str, Any]:
        with self._lock:
            return self._progress.to_dict()

    def _set_progress(self, **changes: Any) -> None:
        with self._lock:
            data = self._progress.to_dict()
            data.pop("progress", None)
            data.update(changes)
            self._progress = ParseProgress(**data)

    def _parse_pdf(self, path: Path) -> int:
        doc = fitz.open(path)
        lines: list[str] = []
        image_count = 0

        metadata = {
            "source_file": path.name,
            "title": doc.metadata.get("title") or path.stem,
            "document_type": "PDF",
            "page_count": doc.page_count,
            "parser": "PyMuPDF",
            "image_ocr_enabled": self.enable_image_ocr,
        }
        lines.extend(self._frontmatter(metadata))

        title = metadata["title"]
        if title:
            lines.append(f"# {title}")
            lines.append("")

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            lines.append(f"## Page {page_number}")
            lines.append("")

            page_lines = self._page_to_markdown(page)
            if not page_lines and self.enable_page_ocr_fallback:
                page_ocr = self._ocr_page(page)
                if page_ocr:
                    page_lines.extend(page_ocr)

            lines.extend(page_lines)
            image_lines, count = self._extract_images(path.stem, page)
            if image_lines:
                lines.extend(image_lines)
                image_count += count

        output_path = self.output_dir / f"{path.stem}.md"
        output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return image_count

    def _frontmatter(self, metadata: dict[str, Any]) -> list[str]:
        raw = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False).strip()
        return ["---", raw, "---", ""]

    def _page_to_markdown(self, page: fitz.Page) -> list[str]:
        blocks = page.get_text("dict").get("blocks", [])
        lines: list[str] = []
        current_paragraph: list[str] = []

        def flush_paragraph() -> None:
            if current_paragraph:
                lines.append(" ".join(current_paragraph).strip())
                lines.append("")
                current_paragraph.clear()

        text_blocks = [block for block in blocks if block.get("type") == 0]
        for block in text_blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(span.get("text", "") for span in spans).strip()
                if not text:
                    continue
                max_size = max((span.get("size", 0.0) for span in spans), default=0.0)
                if self._looks_like_heading(text, max_size):
                    flush_paragraph()
                    level = "###" if max_size >= 14 else "####"
                    lines.append(f"{level} {text}")
                    lines.append("")
                    continue
                current_paragraph.append(text)
            flush_paragraph()
        return lines

    def _looks_like_heading(self, text: str, font_size: float) -> bool:
        if font_size >= 15:
            return True
        if len(text) <= 120 and HEADER_RE.match(text):
            return True
        return False

    def _extract_images(self, doc_stem: str, page: fitz.Page) -> tuple[list[str], int]:
        lines: list[str] = []
        count = 0
        for image_index, image_info in enumerate(page.get_images(full=True), start=1):
            xref = image_info[0]
            extracted = page.parent.extract_image(xref)
            image_bytes = extracted["image"]
            ext = extracted.get("ext", "png")
            asset_name = f"{doc_stem}_page_{page.number + 1}_img_{image_index}.{ext}"
            asset_path = self.assets_dir / asset_name
            asset_path.write_bytes(image_bytes)

            lines.append("<image_summary>")
            lines.append(f"page: {page.number + 1}")
            lines.append(f"path: {asset_path.as_posix()}")
            if self.enable_image_ocr:
                ocr_text = self._ocr_image_bytes(image_bytes)
                if ocr_text:
                    lines.append("ocr_text: |")
                    lines.extend(f"  {line}" for line in ocr_text.splitlines())
            lines.append("</image_summary>")
            lines.append("")
            count += 1
        return lines, count

    def _ocr_image_bytes(self, image_bytes: bytes) -> str:
        try:
            image = Image.open(BytesIO(image_bytes))
            text = pytesseract.image_to_string(image).strip()
            return self._normalize_ocr(text)
        except Exception:
            return ""

    def _ocr_page(self, page: fitz.Page) -> list[str]:
        try:
            pixmap = page.get_pixmap(dpi=200)
            image = Image.open(BytesIO(pixmap.tobytes("png")))
            text = pytesseract.image_to_string(image).strip()
            normalized = self._normalize_ocr(text)
            if not normalized:
                return []
            return [normalized, ""]
        except Exception:
            return []

    def _normalize_ocr(self, text: str) -> str:
        cleaned = "\n".join(line.strip() for line in text.splitlines() if WORD_RE.search(line))
        return cleaned[:4000]
