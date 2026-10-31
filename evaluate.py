import json
from pathlib import Path
from bayyina.chunking import chunk_text
from bayyina.bm25 import BM25Index
from bayyina.hybrid import HybridRetriever
from bayyina.embedder import HashingEmbedder
from bayyina.types import Hit
import numpy as np

def run_evaluation():
    # 1. Load all contracts from data/contracts/
    contracts_path = Path("data/contracts")
    contract_files = list(contracts_path.glob("*.txt"))
    
    chunks = []
    for path in contract_files:
        doc_id = path.stem
        text = path.read_text(encoding="utf-8")
        file_chunks = chunk_text(doc_id, text, max_chars=500, overlap=50)
        chunks.extend(file_chunks)
        
    chunk_map = {c.chunk_id: c for c in chunks}

    # 2. Load public queries from data/queries.jsonl
    queries_path = Path("data/queries.jsonl")
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # 3. Initialize models
    embedder = HashingEmbedder(dim=512, n=3)
    
    bm25 = BM25Index()
    bm25.add(chunks)
    
    hybrid = HybridRetriever(chunks, embedder)

    chunk_texts = [c.text for c in chunks]
    chunk_embs = embedder.embed(chunk_texts)

    def evaluate_model(search_fn, name):
        found = 0
        total = len(queries)
        for q in queries:
            q_text = q.get("query", "")
            relevant_items = q.get("relevant", [])
            
            hits = search_fn(q_text, top_k=3)
            
            matched = False
            for hit in hits:
                hit_doc = str(hit.doc_id).replace(".txt", "").strip()
                for rel in relevant_items:
                    q_doc = str(rel.get("doc_id", "")).replace(".txt", "").strip()
                    if hit_doc != q_doc:
                        continue
                    r_start = rel.get("start")
                    r_end = rel.get("end")
                    if r_start is not None and r_end is not None:
                        # Interval overlap condition: max(start1, start2) < min(end1, end2)
                        if max(hit.start, r_start) < min(hit.end, r_end):
                            matched = True
                            break
                if matched:
                    break
            if matched:
                found += 1
                
        recall = found / total if total > 0 else 0
        print(f"{name} Recall@3: {recall:.4f} ({found}/{total})")
        return recall

    # BM25 Search Wrapper
    def bm25_search(query, top_k=3):
        results = bm25.search(query, top_k=top_k)
        return [
            Hit(chunk_id=chunk_map[cid].chunk_id, doc_id=chunk_map[cid].doc_id, 
                start=chunk_map[cid].start, end=chunk_map[cid].end, text=chunk_map[cid].text, score=score)
            for cid, score in results
        ]

    # Dense-Only Search Wrapper
    def dense_search(query, top_k=3):
        q_emb = embedder.embed([query])[0]
        dots = np.dot(chunk_embs, q_emb)
        sorted_idx = np.argsort(-dots)[:top_k]
        hits = []
        for idx in sorted_idx:
            c = chunks[idx]
            hits.append(Hit(chunk_id=c.chunk_id, doc_id=c.doc_id, start=c.start, end=c.end, text=c.text, score=float(dots[idx])))
        return hits

    print("\n=== Evaluating Retrievers on Public Queries ===")
    evaluate_model(bm25_search, "BM25 Only")
    evaluate_model(dense_search, "Dense Only")
    evaluate_model(lambda q, top_k: hybrid.search(q, top_k=top_k), "Hybrid (RRF)")

if __name__ == "__main__":
    run_evaluation()