"""Citation checks. You write this file."""
from __future__ import annotations

from bayyina.types import Hit

def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    if not isinstance(quote, str) or not quote.strip():
        return False
    if not isinstance(document, str):
        return False
    if not (0 <= start < end <= len(document)):
        return False
        
    span_text = document[start:end]
    
    # Whitespace collapsed: " ".join(s.split())
    collapsed_quote = " ".join(quote.split())
    collapsed_span = " ".join(span_text.split())
    
    return collapsed_quote in collapsed_span


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """[doc_id:start-end] "quote" """
    doc_id = hit.doc_id
    start = hit.start
    end = hit.end
    
    # Whitespace collapse hit text
    collapsed_text = " ".join(hit.text.split())
    
    if len(collapsed_text) > max_quote:
        # Take first max_quote characters
        truncated = collapsed_text[:max_quote]
        # Cut back to the last space inside them (dropping the space) and append…
        last_space = truncated.rfind(' ')
        if last_space != -1:
            quote_part = truncated[:last_space]
        else:
            quote_part = truncated
        quote_str = f"{quote_part}…"
    else:
        quote_str = collapsed_text
        
    return f"[{doc_id}:{start}-{end}] \"{quote_str}\""