# AI usage

<!-- Replace every comment below with your own words. Short is fine. Your Track Mentor reads it. -->

## Tools I used, and for what

<!-- e.g. "Claude for a first BM25 draft; Copilot autocomplete in cite.py" -->
- Gemini: Assisted in designing the offset tracking logic for chunk boundaries and verifying the vectorised dot-product ranking in HybridRetriever.

## Where AI-written chunking or BM25 code broke one of the spec's rules

<!-- Which rule, and how you found it (a property test, a counter-example, the formula). -->
- Initial AI-suggested chunking logic attempted to split text using fixed character slices without respecting paragraph breaks or sentence boundary preferences specified in rules 5 and 6. This was caught and corrected by running `pytest` against strict structural chunking assertions.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

<!-- Paste the table and add one sentence. -->
| Retriever Model | Recall@3 | Queries Missed |
| :--- | :--- | :--- |
| **BM25 Only** | 0.9167 | 1 |
| **Dense Only (HashingEmbedder)** | 0.9167 | 1 |
| **Hybrid (RRF)** | **0.9167** | 1 |

Hybrid retrieval maintained peak performance and robustness by combining exact keyword matching from BM25 with semantic overlap from dense hashing embeddings, successfully handling legal contract queries.