# Text-search design comparison

Owner: Nick.

The comparison will run keyword-only and semantic-only retrieval against the
same questions and processed course pages. The reusable runner is
`course_assistant.text_search.evaluation.compare_retrievers`; it records the
top source, whether the expected supporting source appears in the top results,
and elapsed time. `write_comparison_csv` saves the raw results for the final
README report.

## Final input still needed

Run the comparison after Chase's processed course pages and the team's final
5--10 evaluation questions are available. Each text question needs an expected
supporting document and, when known, page/slide number. Visual-only questions
remain part of the team's overall evaluation but should not be used to judge a
text-only retriever.

## Results template

| Method | Correct source in top results | Average time | Interpretation |
|---|---:|---:|---|
| Keyword (BM25) | Pending final question set | Pending | Pending |
| Text embeddings | Pending final question set | Pending | Pending |

The final README should explain which method succeeds on exact terminology,
which succeeds on paraphrased questions, the latency tradeoff, and why the app
keeps both as hybrid retrieval candidates.
