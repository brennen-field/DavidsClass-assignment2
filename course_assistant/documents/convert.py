"""Convert PPTX/DOCX to PDF with LibreOffice, then render PDF pages to PNG."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

RENDER_DPI = 150
CONVERT_TIMEOUT_SECONDS = 180

# Common install locations when soffice isn't on PATH.
_SOFFICE_CANDIDATES = [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    r"C:\Program Files\LibreOffice\program\soffice.exe",
    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
]

# Include hidden slides so PDF page N is always slide N.
_PPTX_FILTER = 'pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"}}'


class ConversionError(Exception):
    """The file could not be turned into a PDF or rendered."""


def find_soffice() -> str | None:
    """Path to LibreOffice's soffice binary, or None if not installed."""
    configured = os.getenv("SOFFICE_PATH")
    if configured and Path(configured).exists():
        return configured
    found = shutil.which("soffice") or shutil.which("libreoffice")
    if found:
        return found
    return next((c for c in _SOFFICE_CANDIDATES if Path(c).exists()), None)


def convert_to_pdf(source: Path, out_dir: Path) -> Path:
    """Convert a PPTX/DOCX file to PDF in out_dir and return the PDF path."""
    soffice = find_soffice()
    if soffice is None:
        raise ConversionError(
            "LibreOffice is needed to convert PowerPoint/Word files but was not found. "
            "Install it (see README) or export the file to PDF and upload the PDF."
        )

    target = _PPTX_FILTER if source.suffix.lower() == ".pptx" else "pdf"
    # A throwaway profile lets conversion work even while LibreOffice is open elsewhere.
    with tempfile.TemporaryDirectory() as profile:
        cmd = [
            soffice,
            f"-env:UserInstallation={Path(profile).as_uri()}",
            "--headless",
            "--convert-to",
            target,
            "--outdir",
            str(out_dir),
            str(source),
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=CONVERT_TIMEOUT_SECONDS, check=False)
        except subprocess.TimeoutExpired:
            raise ConversionError(f"LibreOffice took too long converting {source.name}.")

    pdf_path = out_dir / f"{source.stem}.pdf"
    if not pdf_path.exists():
        raise ConversionError(f"LibreOffice could not convert {source.name} to PDF.")
    return pdf_path


def render_pages(pdf_path: Path, pages_dir: Path) -> list[Path]:
    """Render each PDF page to pages_dir/page-001.png etc. and return the paths."""
    pages_dir.mkdir(parents=True, exist_ok=True)
    try:
        pdf = fitz.open(pdf_path)
    except Exception as exc:
        raise ConversionError(f"Could not open PDF: {exc}") from exc

    with pdf:
        if pdf.page_count == 0:
            raise ConversionError("The PDF has no pages.")
        paths = []
        for i, page in enumerate(pdf, start=1):
            path = pages_dir / f"page-{i:03d}.png"
            page.get_pixmap(dpi=RENDER_DPI).save(path)
            paths.append(path)
    return paths
