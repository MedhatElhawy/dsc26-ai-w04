"""Reciprocal rank fusion and the hybrid retriever. You write this file."""
from __future__ import annotations

from bayyina.types import Chunk, Embedder, Hit, Reranker
from collections import defaultdict
import numpy as np
from bayyina.bm25 import BM25Index
from bayyina.normalize import normalize_ar

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] += 1.0 / (k + rank)
            
    # Sort by score descending, then ID ascending
    sorted_items = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_items


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None
    ):
        self.chunks = chunks
        self.chunk_map = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.reranker = reranker
        
        # Initialize and populate BM25
        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(chunks)
        
        # Precompute chunk embeddings for dense retrieval
        chunk_texts = [c.text for c in chunks]
        self.chunk_embeddings = self.embedder.embed(chunk_texts) # shape (N, dim)

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        if not query or not query.strip() or not self.chunks:
            return []
            
        norm_query = normalize_ar(query)
        if not norm_query:
            return []
            
        # 1. BM25 search ranking
        bm25_results = self.bm25.search(query, top_k=candidates)
        bm25_ids = [item[0] for item in bm25_results]
        
        # 2. Dense retrieval search ranking (dot product with embedded query, ties by chunk ID)
        q_emb = self.embedder.embed([query])[0] # shape (dim,)
        # Dot products
        dots = np.dot(self.chunk_embeddings, q_emb)
        
        # Create list of (chunk_id, dot_product) for sorting
        dense_scores = [(self.chunks[i].chunk_id, float(dots[i])) for i in range(len(self.chunks))]
        # Sort by dot product descending, then chunk ID ascending
        dense_scores.sort(key=lambda x: (-x[1], x[0]))
        dense_ids = [item[0] for item in dense_scores[:candidates]]
        
        # 3. Fuse with RRF
        fused = rrf([bm25_ids, dense_ids], k=60)
        top_fused_ids = [item[0] for item in fused[:candidates]]
        
        # Build initial Hits
        hits = []
        for cid in top_fused_ids:
            score_dict_val = dict(fused)[cid]
            chunk = self.chunk_map[cid]
            hits.append(Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                start=chunk.start,
                end=chunk.end,
                text=chunk.text,
                score=score_dict_val
            ))
            
        # 4. Optional Re-ranker hook
        if self.reranker is not None and hits:
            # Reranker receives query and list of hits, returns scored hits or scores
            reranked_hits = self.reranker.rerank(query, hits)
            # Sort by reranker score descending, ties by chunk ID ascending
            reranked_hits.sort(key=lambda x: (-x.score, x.chunk_id))
            hits = reranked_hits[:top_k]
        else:
            # Sort RRF hits by score descending, ties by chunk ID ascending
            hits.sort(key=lambda x: (-x.score, x.chunk_id))
            hits = hits[:top_k]
            
        return hits