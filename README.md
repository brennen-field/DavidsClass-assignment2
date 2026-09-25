# Course Assistant

MBAX 6418 Assignment 2. A Python app that answers questions about course
materials and generates practice quizzes, using hybrid RAG (keyword search,
text embeddings, and slide-image embeddings, combined and reranked) with the
class services.

> Work in progress. Sections below get filled in by each owner.

## Setup

### 1. Install the prerequisites

- **Python 3.12.** Newer versions may not have wheels for every dependency yet.
- **LibreOffice**, needed only to upload PowerPoint (`.pptx`) or Word (`.docx`) files directly:

  | OS | Install |
  |---|---|
  | Windows | Download from [libreoffice.org](https://www.libreoffice.org/download/) and run the installer |
  | macOS | `brew install --cask libreoffice`, or download from libreoffice.org |
  | Linux | Your package manager, e.g. `sudo apt install libreoffice` or `sudo pacman -S libreoffice-fresh` |

  The app finds LibreOffice on your PATH or in its default install folder. If it's somewhere else,
  set `SOFFICE_PATH` in `.env` to the full path of `soffice` (`soffice.exe` on Windows).

### 2. Install and launch

```bash
git clone https://github.com/brennen-field/DavidsClass-assignment2.git
cd DavidsClass-assignment2
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # Windows: copy .env.example .env
python app.py
```

Then open http://127.0.0.1:7860 in your browser.

### 3. Add the API key and endpoints

Open `.env` and replace the placeholder values with the class service key and endpoint URLs
(ask the team for them). `.env` is listed in `.gitignore`, so git won't commit it. Never put
real keys in `.env.example`, code, screenshots, or chat messages.

### 4. Run the tests

```bash
pytest
```

The tests don't call the class services (those calls are replaced with fakes). The PPTX/DOCX
conversion tests are skipped automatically if LibreOffice isn't installed.

## Managing documents

Use the **Documents** tab to upload, preview, and remove course files.

| Format | How it's handled |
|---|---|
| PDF (`.pdf`) | Used as-is |
| PowerPoint (`.pptx`) | Converted to PDF with LibreOffice; hidden slides are kept so page N is always slide N |
| Word (`.docx`) | Converted to PDF with LibreOffice |

Each page or slide is saved as a PNG image (150 DPI) together with its text, the document name,
and the page/slide number. These are what text search and visual search index. Everything is
stored locally in `data/` (not committed).

- **Duplicates:** files are identified by their content (SHA-256 hash). Uploading the same file
  again, even under a different name, is skipped. Uploading a *different* file with the same name
  as a loaded one is refused; remove the old one first, so citations are never ambiguous.
- **Removing** a document deletes its files and its entries in the text and visual search indexes.
  If an index can't be cleared, the document stays listed so you can retry.
- **Text extraction** uses the class document-parsing service. If it's unavailable, the app falls
  back to local extraction (PyMuPDF) and shows `local` in the "Text from" column.
- **No LibreOffice?** Export the deck to PDF in PowerPoint (File → Export → PDF) or Google Slides
  (File → Download → PDF) and upload the PDF instead.
- **Old formats** (`.ppt`, `.doc`) aren't accepted. Re-save them as `.pptx`/`.docx` or export to PDF.
- **Speaker notes** aren't included, only what's visible on the slide.

## Project layout

| Folder | Owner | Part |
|---|---|---|
| `course_assistant/documents/` | Chase | Upload, conversion, page text + images, duplicates, removal |
| `course_assistant/text_search/` | Nick | Chunking, BM25, text embeddings |
| `course_assistant/visual_search/` | Shrihari | Image embeddings, combining, reranking |
| `course_assistant/answers/` | Preston | LLM answers with sources |
| `course_assistant/quiz/` | Brennen | Quizzes; `app.py` shell |
| `docs/` | Everyone | Handoff specs, diagram |

## Team workflow

- Work on your own branch (e.g. `chase/documents`), open a PR, get one review, then merge.
- Never push to `main` directly. Never commit `.env` or keys.
- Review rotation: Chase → Nick → Shrihari → Preston → Brennen → Chase.
