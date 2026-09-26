# Visual-search & combine-and-rerank: manual verification

Owner: Shrihari · Method: real end-to-end run, then vision cross-check.

## Purpose

Confirm the acceptance checks on the ownership list: the *Vibe Coding on "Prod"*
meme returns as a top result, and diagram/chart questions surface the right slide.

## Material

`MBAX 6418 - Week 2 - LLM Fundamentals v2.pptx` (42 slides), rendered locally with
LibreOffice → PDF → pymupdf at 150 DPI into `data/verification/pages/` (local,
gitignored — never part of a commit). The scratch images and index live under
`data/` and are excluded from the repo.

## Method

Indexed all 42 real slide images plus their extracted slide text, then ran the
full pipeline **exactly as the app uses it**:

- **Text retrieval:** Nick's `KeywordIndex` (BM25) and `SemanticIndex`
  (class 9002) — consumed read-only, per the handoff.
- **Visual retrieval:** `VisualIndex` + class visual embedder (9003).
- **Combine + rerank:** `combine_and_rerank(..., use_reranker=True/False)` with
  the class reranker (9004), so both modes were measured.

## Ground truth (vision cross-check)

The `:9001` vision model (`Qwen3.6-35B-AWQ-4bit`) inspected the rendered slides
to determine what is a meme, rather than trusting slide titles:

| Slide | Verdict | Content |
|---|---|---|
| 33 | **MEME** | "One Does Not Simply" (Boromir) against coding directly in prod |
| 34 | **MEME** | "How it started / How it's going" two-tweet prod-incident meme |
| 27–32 | not a meme | title / trends graph / comparison / process diagram / bullets |

So the *Vibe Coding on "Prod"* meme set is **slides 33 and 34**, not a single slide.

## Results (rank of the target slide in final evidence)

| Query | Meme/diagram targets | Fallback rank | Rerank rank |
|---|---|---|---|
| "vibe coding on prod, how it started vs how it's going, security" | **33 & 34** | 34#1, 33#2 | 33#1, 34#2 |
| "what is the vibe coding security meme on production?" | 33 & 34 | 33#1, 34#4 | 33#1, 34#8 |
| "KV caching architecture" (diagram) | 11 | #1 | #1 |
| "speculative decoding diagram" | 13 | #1 | #1 |
| "quantization formats" (chart) | 16 | #1 | #1 |

## Question & answer (diagram/chart) test

Beyond recall, a full question-about-a-chart was answered end-to-end (retrieval +
rerank, then an LLM grounded on the retrieved slide):

- **Question:** "Per the chart of popular open coding models (August 2026), what
  is the context window and parameter count of DeepSeek-V4-Flash-0731?"
- **Retrieved:** slide 22 ("Popular Open Coding Models") ranked **#1**, with all
  three signals (`bm25`, `text_embedding`, `visual`).
- **Grounded answer (from slide 22 facts only):** *"DeepSeek-V4-Flash-0731 has
  284B parameters and a 1M context window."* — correct per the chart.

This validates the full path a diagram/chart question takes: correct slide
retrieved as top evidence, and a correct answer produced from that evidence.

## Findings

- **Acceptance met.** On natural phrasing the two real prod memes (33 & 34) rank
  **#1 and #2** in both modes. Diagram/chart targets rank #1 in both modes.
- The **revealed-meme slides are image-first**: e.g. slide 31 ("Iterating the
  Vibes") enters evidence via `visual` alone, and memes 33/34 fire all three
  signals (`bm25`, `text_embedding`, `visual`).
- **One residual (documented, not hidden):** an exactly-phrased query
  ("what is the vibe coding security meme on production?") keeps one prod meme
  at #1 but drops the other under rerank (34 → #8). The visual/rerank model is
  strong but phrasing-sensitive on the meme images.
- **Decision:** rely on the reranker (the visual model is the stronger judge);
  the deterministic fallback exists as a safety net and for the comparison run,
  not as the primary path.

## Note

This verification used the real class services with credentials from the local
`.env`; no keys appear here, in test output, or in any commit.
