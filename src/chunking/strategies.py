from __future__ import annotations

def fixed_size_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    step = chunk_size - overlap
    if step < 1:
        step = 1

    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text:
            chunks.append(chunk_text)
    return chunks

def sliding_window_chunks(text: str, window_size: int, step_size:int) -> list[str]:
    words = text.split()
    if step_size < 1:
        step_size = 1

    chunks = []
    for start in range(0, len(words), step_size):
        chunk_words = words[start:start + window_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    return chunks

def semantic_chunks(text: str, min_tokens: int, max_tokens: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        candidate = f"{current_chunk}\n\n{para}".strip() if current_chunk else para
        if len(candidate.split()) > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)

            if len(para.split()) > max_tokens:
                chunks.extend(fixed_size_chunks(para, max_tokens, min_tokens))
            else:
                current_chunk = para
        else:
            current_chunk = candidate
        
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk)

    return chunks

