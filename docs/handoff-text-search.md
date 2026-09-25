# Handoff: Text search -> Combine and rerank

Owner: Nick. Code: `course_assistant/text_search/`.

Text search accepts any page record with the fields in Chase's proposed `Page`
contract: `doc_id`, `doc_name`, `page_number`, `text`, `image_path`, and
`source_format`. Empty page text is skipped because the visual index owns
image-only retrieval.

## Keyword results

`KeywordIndex.search(query, top_k=5)` returns `SearchResult` records ordered by
descending BM25 score. Each result contains:

- `chunk_id`: stable identifier for the passage
- `doc_id`: stable document identifier used for replacement and deletion
- `doc_name`: original filename for citations
- `page_number`: 1-based page or slide number
- `text`: the actual source passage
- `image_path`: rendered source page for downstream evidence display
- `source_format`: `pdf`, `pptx`, or `docx`
- `score`: method-specific relevance score
- `search_method`: currently `bm25`

Downstream code should not compare raw scores from different retrieval methods
as though they share a scale. Combine candidates by `chunk_id`, retain the
source fields, and let the reranker produce the final cross-method ordering.

## Document-store hooks

`connect_document_store(store, index)` registers:

- `index.index_pages` with `store.on_document_added(...)`
- `index.delete_document` with `store.register_index_deleter(...)`

Deletion is idempotent. Re-indexing pages for an existing `doc_id` replaces its
old chunks rather than creating duplicates.
