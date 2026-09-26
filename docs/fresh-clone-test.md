# Fresh-clone test

Owner: Shrihari · Run on a clean copy, following the README Setup literally.

## Scope

Verified that a brand-new teammate can set up this repo using only the README,
and recorded what is missing or inaccurate. The clone source was **`origin/main`**
(`276a663`) — the published repo a real cloner/instructor/grader would get, not
any feature branch.

## Steps and outcomes

| README step | Result | Notes |
|---|---|---|
| `git clone https://github.com/brennen-field/DavidsClass-assignment2.git` | ✅ | |
| "Requires Python 3.12" | ⚠️ | Plain `python` on this machine is **3.11.16** and everything worked; the stated 3.12 requirement is not enforced. Claim is stale/off-by-one. |
| `python -m venv .venv` | ✅ | |
| `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`) | ✅ | |
| `pip install -r requirements.txt` | ✅ | gradio, chromadb, pymupdf, bm25s, langchain-text-splitters, requests, python-dotenv all import OK. |
| `cp .env.example .env` | ✅ | `.env` correctly gitignored. |
| "…then fill in real values; never commit .env" | ⚠️ | Under-specified — see Finding below. |
| `python app.py` | ✅ | Launches; Gradio serves at `http://127.0.0.1:7860`. No key needed for the placeholder panels. |
| `pytest` | ✅ | **23 passed** on `origin/main`. |

## Finding: `.env` filling is under-specified (for Brennen, repo-structure owner)

The README says only "fill in real values; never commit .env" but never says
**which values** or **where they come from**. On `origin/main`, `.env.example`
contains:

```text
CLASS_API_KEY=
LLM_URL=https://dobolyi.com:PORT
TEXT_EMBED_URL=http://dobolyi.com:9002/v2/embed
TEXT_EMBED_MODEL=nvidia/Nemotron-3-Embed-1B-BF16
VISUAL_EMBED_URL=https://dobolyi.com:PORT
RERANK_URL=https://dobolyi.com:PORT
PARSE_URL=https://dobolyi.com:PORT
```

A fresh user cannot tell that they must set `CLASS_API_KEY`, nor where the real
ports for `LLM_URL`/`VISUAL_EMBED_URL`/`RERANK_URL`/`PARSE_URL` come from (they
are class-service endpoints "from the team"). Suggested fix for whoever owns the
Setup section: add a short **Configure** note listing the required keys and that
the endpoint ports come from the team.

## What was NOT covered (out of scope for this run)

- This is a repo-`main` setup check only. The `visual_search` module lives on
  the `shrihari/visual-search-and-reranking` branch and is not part of `main`.
- LibreOffice (a stated requirement) is present on this machine, so PPTX
  upload/conversion paths were available; a user without it would hit that
  requirement during PPTX upload.
- Real retrieval/embedding calls require a populated `.env` with a valid
  `CLASS_API_KEY`; this run launched the app without one (panels are placeholders).
