"""BM25, written by hand. You write this file (no search libraries)."""
from __future__ import annotations

from bayyina.types import Chunk
import math
from collections import Counter
from bayyina.normalize import tokenize

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.tokenized_chunks: list[list[str]] = []
        self.doc_lengths: list[int] = []
        self.avg_dl: float = 0.0
        self.df: Counter = Counter()
        self.N: int = 0

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.N = len(chunks)
        if self.N == 0:
            return
        
        self.tokenized_chunks = [tokenize(c.text) for c in chunks]
        self.doc_lengths = [len(tc) for tc in self.tokenized_chunks]
        self.avg_dl = sum(self.doc_lengths) / self.N if self.N > 0 else 0.0
        
        self.df = Counter()
        for tc in self.tokenized_chunks:
            unique_terms = set(tc)
            for term in unique_terms:
                self.df[term] += 1

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""

        if not query or not self.chunks or self.N == 0:
            return []
            
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
            
        # Distinct query terms count once for frequency in query, but for BM25 matching we score terms present
        q_terms = set(q_tokens)
        scores = [0.0] * self.N
        
        for term in q_terms:
            if term not in self.df:
                continue
            
            df_t = self.df[term]
            # IDF formula: ln(1 + (N - df + 0.5) / (df + 0.5))
            idf = math.log(1.0 + (self.N - df_t + 0.5) / (df_t + 0.5))
            
            for idx, tc in enumerate(self.tokenized_chunks):
                tf = tc.count(term)
                if tf == 0:
                    continue
                
                dl = self.doc_lengths[idx]
                # BM25 term score formula
                norm_dl = 1.0 - self.b + self.b * (dl / self.avg_dl) if self.avg_dl > 0 else 1.0
                term_score = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm_dl)
                scores[idx] += term_score
                
        # Filter scores > 0, sort by score descending, then chunk ID ascending
        results = []
        for idx, score in enumerate(scores):
            if score > 0.0:
                results.append((self.chunks[idx].chunk_id, score))
                
        # Sort: score descending (-score), chunk ID ascending (chunk_id)
        results.sort(key=lambda x: (-x[1], x[0]))
        
        return results[:top_k]