"""The document collection: add, list, get pages, and remove documents.

On disk (under DATA_DIR):

    catalog.json                      one entry per document
    documents/<doc_id>/original.pptx  the uploaded file
    documents/<doc_id>/document.pdf   PDF used for rendering (converted if needed)
    documents/<doc_id>/pages/page-001.png
    documents/<doc_id>/pages.json     per-page text
"""

import hashlib
import json
import logging
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from course_assistant import config
from course_assistant.documents.convert import ConversionError, convert_to_pdf, render_pages
from course_assistant.documents.models import Document, Page
from course_assistant.documents.parsing import extract_page_texts

log = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".pdf": "pdf", ".pptx": "pptx", ".docx": "docx"}

AddedListener = Callable[[list[Page]], None]
IndexDeleter = Callable[[str], None]


@dataclass
class AddResult:
    status: str  # "added", "duplicate", or "error"
    message: str
    document: Document | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class RemoveResult:
    ok: bool
    message: str


def file_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class DocumentStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.docs_dir = self.data_dir / "documents"
        self.catalog_path = self.data_dir / "catalog.json"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._listeners: list[AddedListener] = []
        self._deleters: list[IndexDeleter] = []

    # ---- hooks for text and visual search ----

    def on_document_added(self, listener: AddedListener) -> None:
        self._listeners.append(listener)

    def register_index_deleter(self, deleter: IndexDeleter) -> None:
        self._deleters.append(deleter)

    # ---- reading ----

    def list_documents(self) -> list[Document]:
        docs = [Document(**entry) for entry in self._read_catalog().values()]
        return sorted(docs, key=lambda d: d.doc_name.lower())

    def get_document(self, doc_id: str) -> Document | None:
        entry = self._read_catalog().get(doc_id)
        return Document(**entry) if entry else None

    def get_pages(self, doc_id: str) -> list[Page]:
        doc = self.get_document(doc_id)
        if doc is None:
            return []
        doc_dir = self.docs_dir / doc_id
        texts = json.loads((doc_dir / "pages.json").read_text(encoding="utf-8"))
        return [
            Page(
                doc_id=doc_id,
                doc_name=doc.doc_name,
                page_number=i,
                text=text,
                image_path=str(doc_dir / "pages" / f"page-{i:03d}.png"),
                source_format=doc.source_format,
            )
            for i, text in enumerate(texts, start=1)
        ]

    # ---- adding ----

    def add_file(self, path: Path, doc_name: str | None = None) -> AddResult:
        path = Path(path)
        doc_name = doc_name or path.name
        suffix = Path(doc_name).suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_FORMATS))
            return AddResult("error", f"{doc_name}: unsupported file type. Supported: {allowed}.")

        data = path.read_bytes()
        if not data:
            return AddResult("error", f"{doc_name}: the file is empty.")
        doc_id = file_id(data)

        with self._lock:
            catalog = self._read_catalog()
            if doc_id in catalog:
                existing = catalog[doc_id]["doc_name"]
                return AddResult(
                    "duplicate",
                    f"{doc_name}: already loaded as {existing}. Skipped.",
                    Document(**catalog[doc_id]),
                )
            if any(entry["doc_name"] == doc_name for entry in catalog.values()):
                return AddResult(
                    "error",
                    f"{doc_name}: a different file with this name is already loaded. "
                    "Remove it first, or rename the new file.",
                )

            try:
                doc = self._process(data, doc_id, doc_name, SUPPORTED_FORMATS[suffix])
            except ConversionError as exc:
                return AddResult("error", f"{doc_name}: {exc}")

            catalog[doc_id] = doc.to_dict()
            self._write_catalog(catalog)

        warnings = self._notify_added(doc)
        return AddResult("added", f"{doc_name}: added ({doc.page_count} pages).", doc, warnings)

    def _process(self, data: bytes, doc_id: str, doc_name: str, source_format: str) -> Document:
        """Build the document folder in a temp location, then move it into place."""
        work_dir = self.docs_dir / f".tmp-{doc_id}-{uuid.uuid4().hex[:8]}"
        work_dir.mkdir()
        try:
            original = work_dir / f"original.{source_format}"
            original.write_bytes(data)

            if source_format == "pdf":
                pdf_path = original
            else:
                converted = convert_to_pdf(original, work_dir)
                pdf_path = converted.rename(work_dir / "document.pdf")

            image_paths = render_pages(pdf_path, work_dir / "pages")
            texts, text_source = extract_page_texts(pdf_path, len(image_paths))
            (work_dir / "pages.json").write_text(
                json.dumps(texts, ensure_ascii=False, indent=1), encoding="utf-8"
            )

            final_dir = self.docs_dir / doc_id
            if final_dir.exists():  # leftover from an interrupted removal
                shutil.rmtree(final_dir)
            work_dir.rename(final_dir)
        except BaseException:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise

        return Document(
            doc_id=doc_id,
            doc_name=doc_name,
            source_format=source_format,
            page_count=len(image_paths),
            added_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            text_source=text_source,
        )

    def _notify_added(self, doc: Document) -> list[str]:
        if not self._listeners:
            return []
        pages = self.get_pages(doc.doc_id)
        warnings = []
        for listener in self._listeners:
            try:
                listener(pages)
            except Exception as exc:
                log.exception("Indexing failed for %s", doc.doc_name)
                warnings.append(
                    f"{doc.doc_name}: indexing step failed ({type(exc).__name__}); "
                    "it may not be searchable."
                )
        return warnings

    # ---- removing ----

    def remove(self, doc_id: str) -> RemoveResult:
        with self._lock:
            catalog = self._read_catalog()
            if doc_id not in catalog:
                return RemoveResult(False, "That document is not loaded.")
            doc_name = catalog[doc_id]["doc_name"]

            # Clear the search indexes first. If any fail, keep the document so the
            # user can retry, instead of leaving searchable entries for a hidden file.
            for deleter in self._deleters:
                try:
                    deleter(doc_id)
                except Exception as exc:
                    log.exception("Index delete failed for %s", doc_name)
                    return RemoveResult(
                        False,
                        f"{doc_name}: could not remove it from a search index "
                        f"({type(exc).__name__}). Nothing was removed; try again.",
                    )

            del catalog[doc_id]
            self._write_catalog(catalog)
            shutil.rmtree(self.docs_dir / doc_id, ignore_errors=True)

        return RemoveResult(True, f"{doc_name}: removed.")

    # ---- catalog file ----

    def _read_catalog(self) -> dict:
        if not self.catalog_path.exists():
            return {}
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _write_catalog(self, catalog: dict) -> None:
        tmp = self.catalog_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.catalog_path)


_store: DocumentStore | None = None


def get_store() -> DocumentStore:
    """The app-wide document store, stored under DATA_DIR from .env."""
    global _store
    if _store is None:
        _store = DocumentStore(config.DATA_DIR)
    return _store
