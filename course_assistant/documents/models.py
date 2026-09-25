"""Records passed from the documents part to text and visual search.

See docs/handoff-documents.md for how other parts use these.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Page:
    """One page of a PDF or one slide of a deck."""

    doc_id: str  # stable ID for the document (hash of the file's bytes)
    doc_name: str  # original file name, e.g. "Week 2 - Vibe Coding.pptx"
    page_number: int  # 1-based page/slide number, as shown in the original
    text: str  # text on this page ("" if the page is only an image)
    image_path: str  # absolute path to a PNG of the rendered page
    source_format: str  # "pdf", "pptx", or "docx"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Document:
    """One uploaded file, as listed in the document manager."""

    doc_id: str
    doc_name: str
    source_format: str
    page_count: int
    added_at: str  # ISO 8601 timestamp (UTC)
    text_source: str  # "class-service" or "local" (which text extractor was used)

    def to_dict(self) -> dict:
        return asdict(self)
