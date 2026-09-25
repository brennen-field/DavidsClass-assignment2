# Handoff: Text search -> Combine and rerank

Owner: Nick. Code: `course_assistant/text_search/`.

Text search accepts any page record with the fields in Chase's proposed `Page`
contract: `doc_id`, `doc_name`, `page_number`, `text`, `image_path`, and
`source_format`. Empty page text is skipped because the visual index owns
image-only retrieval.

## Search results

Both `KeywordIndex.search(query, top_k=5)` and
`SemanticIndex.search(query, top_k=5)` return `SearchResult` records ordered by
descending method-specific score. Each result contains:

- `chunk_id`: stable identifier for the passage
- `doc_id`: stable document identifier used for replacement and deletion
- `doc_name`: original filename for citations
- `page_number`: 1-based page or slide number
- `text`: the actual source passage
- `image_path`: rendered source page for downstream evidence display
- `source_format`: `pdf`, `pptx`, or `docx`
- `score`: method-specific relevance score
- `search_method`: `bm25` or `text_embedding`

Downstream code should not compare raw scores from different retrieval methods
as though they share a scale. Combine candidates by `chunk_id`, retain the
source fields, and let the reranker produce the final cross-method ordering.

## Text embeddings

`TextEmbeddingClient.from_environment()` reads `CLASS_API_KEY`,
`TEXT_EMBED_URL`, and `TEXT_EMBED_MODEL` from the ignored local `.env` file.
It sends document chunks with `input_type=document` and search questions with
`input_type=query` to the class `/v2/embed` endpoint. Credentials are never
logged or included in errors or object representations.

`SemanticIndex.index_pages(...)` embeds the same source-preserving chunks used
by keyword search. It validates the service response before replacing an
existing document, so a temporary model outage does not erase a working index.
Its cosine-similarity results use the same `SearchResult` contract as BM25.

## Document-store hooks

Call `connect_document_store(store, index)` for each text index. It registers:

- `index.index_pages` with `store.on_document_added(...)`
- `index.delete_document` with `store.register_index_deleter(...)`

Deletion is idempotent. Re-indexing pages for an existing `doc_id` replaces its
old chunks rather than creating duplicates.
