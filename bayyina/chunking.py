"""Offset-preserving chunker. You write this file. See README.md for the seven rules."""
from __future__ import annotations

from bayyina.types import Chunk
import re

def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """    
    if max_chars <= 0 or not (0 <= overlap < max_chars):
        raise ValueError("Invalid max_chars or overlap parameters.")
    
    if not text or not text.strip():
        return []
    
    chunks: list[Chunk] = []
    n = len(text)
    s = 0
    chunk_idx = 0
    
    while s < n:
        # Skip leading whitespace for the new chunk, but keep track if we started mid-word
        # Rule 6: A chunk may start mid-word only when previous chunk ended with mid-word cut.
        # Our cut logic handles start offsets naturally.
        
        # Check if we can reach the end
        if n - s <= max_chars:
            end = n
            # Strip trailing whitespace for chunk text
            chunk_raw = text[s:end]
            stripped_raw = chunk_raw.rstrip()
            actual_end = s + len(stripped_raw)
            
            if len(stripped_raw) > 0:
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}#{chunk_idx}",
                    doc_id=doc_id,
                    start=s,
                    end=actual_end,
                    text=stripped_raw
                ))
            break
            
        # Window is text[s : s + max_chars]
        window_end = min(s + max_chars, n)
        window = text[s:window_end]
        
        # Search for cut points in the second half of the window: max_chars // 2 <= c <= max_chars
        half = max_chars // 2
        best_c = -1
        
        # Priority 1: Just after the last paragraph break \n\n inside the second half
        # \n\n lies inside window if present at c-2, c-1
        # Let's scan backwards or forwards in range [half, max_chars]
        best_priority = -1 # 3: \n\n, 2: sentence end / newline, 1: space, 0: hard cut
        
        sentence_enders = {'.', '!', '?', '؟', '۔', '\n'}
        
        for c in range(min(window_end - s, max_chars), half - 1, -1):
            if s + c > n:
                continue
            
            # Check Priority 3: \n\n ending at c
            if c >= 2 and text[s + c - 2:s + c] == '\n\n':
                best_c = c
                best_priority = 3
                break
            
            # Check Priority 2: Sentence end or single \n ending at c-1
            if c >= 1 and text[s + c - 1] in sentence_enders:
                if best_priority < 2:
                    best_c = c
                    best_priority = 2
            
            # Check Priority 1: Space ending at c-1
            elif c >= 1 and text[s + c - 1].isspace():
                if best_priority < 1:
                    best_c = c
                    best_priority = 1
                    
        if best_c == -1:
            # Fallback to hard cut at max_chars
            best_c = min(max_chars, n - s)
            
        end = s + best_c
        chunk_raw = text[s:end]
        stripped_raw = chunk_raw.rstrip()
        actual_end = s + len(stripped_raw)
        
        if len(stripped_raw) > 0:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#{chunk_idx}",
                doc_id=doc_id,
                start=s,
                end=actual_end,
                text=stripped_raw
            ))
            chunk_idx += 1
            
        # Determine next start `s` ensuring overlap constraint (at most `overlap`)
        # Next start should be end - overlap, but we want to avoid getting stuck or violating constraints.
        # Rule 4: consecutive chunks overlap by at most overlap characters.
        # Usually, next start is max(0, end - overlap). To avoid infinite loops if best_c is small, ensure s advances.
        next_s = end - overlap
        if next_s <= s:
            next_s = s + 1
        s = next_s
        
        # Skip any leading whitespace for the next chunk start to keep clean words, 
        # unless it was a mid-word cut.
        # A chunk ended with mid-word cut when characters on both sides of its end are not whitespace.
        if end < n and not text[end - 1].isspace() and not text[end].isspace():
            # Mid-word cut: next chunk starts exactly at `end` without skipping whitespace
            s = end
        else:
            # Normal cut: skip whitespace to start at the beginning of a word
            while s < n and text[s].isspace():
                s += 1
                
    return chunks