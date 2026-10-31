# Retrieval evaluation

<!-- Read by your Track Mentor, not graded by tests. Replace the comments. -->

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 | 1 |
| Dense only (HashingEmbedder) | 0.9167 | 1 |
| Hybrid (RRF) | 0.9167 | 1 |

<!-- Two or three sentences: what the misses have in common, and what you would try next
     (a real embedding model, contextual chunk headers, a re-ranker, different chunk size). -->

The single missed query across all models involves complex semantic paraphrasing where exact statutory keywords are absent and character n-gram hashing fails to capture deep conceptual equivalence. To improve retrieval on such edge cases next, I would try integrating a real transformer-based embedding model (such as a multilingual sentence encoder) alongside a cross-encoder re-ranker to capture deeper semantic relationships and boost ranking precision.
