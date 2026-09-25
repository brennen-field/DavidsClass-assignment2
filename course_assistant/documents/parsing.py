"""Extract per-page text, using the class parsing service with a local fallback."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from course_assistant import config

log = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """The class parsing service is not configured, unreachable, or returned bad data."""


def parse_with_class_service(pdf_path: Path) -> list[str]:
    """Return one text string per page from the class document-parsing service.

    TODO: implement once we have the parsing service's documented request and
    response format. Until then this reports the service as unavailable, so the
    local fallback is used.
    """
    if not config.PARSE_URL or "PORT" in config.PARSE_URL:
        raise ServiceUnavailable("PARSE_URL is not configured")
    raise ServiceUnavailable("class parsing client not implemented yet")


def parse_locally(pdf_path: Path) -> list[str]:
    """Return one text string per page using PyMuPDF's built-in text extraction."""
    with fitz.open(pdf_path) as pdf:
        return [page.get_text().strip() for page in pdf]


def extract_page_texts(pdf_path: Path, expected_pages: int) -> tuple[list[str], str]:
    """Return (texts, source), where source is "class-service" or "local"."""
    try:
        texts = parse_with_class_service(pdf_path)
        if len(texts) != expected_pages:
            raise ServiceUnavailable(
                f"service returned {len(texts)} pages, expected {expected_pages}"
            )
        return texts, "class-service"
    except ServiceUnavailable as exc:
        # Never include keys or full URLs in this message.
        log.warning("Parsing service unavailable (%s); using local text extraction.", exc)
        return parse_locally(pdf_path), "local"
