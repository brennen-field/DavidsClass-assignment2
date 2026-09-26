# Handoff: Visual search + combine-and-rerank -> Answers & Quiz

Owner: Shrihari. Code: `course_assistant/visual_search/`.

Consumes Chase's page images (rendered slides) and Nick's keyword + text-embedding
`SearchResult`s, and produces the **final ranked evidence** that Answers (Preston)
and Quiz (Brennen) ground their LLM answers on.

## Public entry point

```python
from course_assistant.visual_search import combine_and_rerank

evidence = combine_and_rerank(
    query,
    keyword_results,        # list[SearchResult]   from Nick's KeywordIndex
    text_embedding_results, # list[SearchResult]   from Nick's SemanticIndex / ChromaSemanticIndex
    visual_results,         # list[VisualHit]      from the visual index (below)
    use_reranker=True,      # on/off switch for the comparison run
    reranker=None,          # RerankerClient.from_environment() when use_reranker
)
```

`evidence` is a `list[EvidenceItem]`, best first, ready to feed an LLM answer with
citations.

## EvidenceItem (the record you receive)

| Field | Meaning |
|---|---|
| `chunk_id` | stable text-passage id (empty `""` for image-only pages) |
| `doc_id` | stable document id (used for replacement/deletion) |
| `doc_name` | original filename for citations |
| `page_number` | 1-based page/slide number |
| `text` | the source passage; **may be empty for image-only slides** |
| `image_path` | rendered page image for evidence display |
| `source_format` | `pdf`, `pptx`, `docx` |
| `rerank_score` | final order score: reranker relevance when reranking is on; otherwise the best per-method retrieval score |
| `retrieved_by` | tuple of methods that found it, e.g. `("bm25", "text_embedding", "visual")` |

**Display guidance:** when `text` is empty, render `image_path` (visual search owns
image-only slides). When `text` is present you may show it alongside `image_path`.

## Visual index

`VisualIndex` (in-memory) and `ChromaVisualIndex` (persistent, at
`DATA_DIR/visual_vectors`) run on the class visual embedder (`9003`,
`Qwen/Qwen3-VL-Embedding-2B`). Their `search(query, top_k)` returns
`VisualHit` records (doc_id, doc_name, page_number, image_path, source_format,
score). Both expose `index_pages`/`delete_document` and connect to Chase's
document store through `connect_document_store` (re-exported).

Images are sent to the service as inline base64 data URLs. Note: this embedding
model accepts only `input_type="default"` (not `document`/`query`).

## Combining

Candidates from keyword + text embedding are merged by `chunk_id` (source fields
and every method that found them are retained). A `VisualHit` attributes
`"visual"` to any text candidate on the same `doc_id`/`page_number`; a visual hit
with no matching text candidate becomes an image-only `EvidenceItem`. Raw scores
from different methods are never compared against each other.

## Reranking

Reranking uses the class multimodal reranker (`9004`,
`Qwen/Qwen3-VL-Reranker-2B`), which scores each candidate on text **and** its
page image against the query. `use_reranker=True` (default) orders by rerank
score; with reranking off (for the comparison run), the order is deterministic:
most methods found, then best per-method score. A rerank-service outage degrades
to that fallback order rather than failing retrieval.

## Configuration

Requires the local `.env` (`CLASS_API_KEY` only). Endpoints are pinned in
`.env.example`:

```text
VISUAL_EMBED_URL=http://dobolyi.com:9003/v2/embed
VISUAL_EMBED_MODEL=Qwen/Qwen3-VL-Embedding-2B
RERANK_URL=http://dobolyi.com:9004/v2/rerank
RERANK_MODEL=Qwen/Qwen3-VL-Reranker-2B
```

No API key is committed or shown in errors/`repr`. Tests are fully mocked and
need no key (CI-safe).

## Verification

See `docs/visual-search-verification.md` for a real end-to-end run on the Week-2
deck (vision cross-checked which slides are the "Vibe Coding on Prod" memes).
