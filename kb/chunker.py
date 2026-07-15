"""Sentence-aligned chunking.

A "chunk" is the unit that gets embedded, indexed, and returned as a search
result. Rules (from plan.txt):

  - Chunk boundaries are COMPLETE sentences, never mid-sentence. We use
    pysbd (a rule-based sentence boundary detector) rather than splitting
    on periods, so "Dr.", "U.S.", "e.g." don't cause false splits.
  - Chunks are sliding windows of `chunk_sentences` sentences that advance
    by (chunk_sentences - overlap), giving consecutive chunks an overlap of
    `chunk_overlap_sentences`. Overlap ensures an idea straddling a chunk
    boundary appears intact in at least one chunk.
  - Chunks never cross page boundaries, so every chunk has one page number
    for citation ("taxes.pdf, p. 12").
"""

import pysbd

# One shared segmenter. English rules work acceptably on most Latin-script
# text; clean=False keeps the original text untouched.
_SEGMENTER = pysbd.Segmenter(language="en", clean=False)

# Chunks shorter than this (in characters) carry no useful signal
# (page numbers, stray headers) and are dropped.
_MIN_CHUNK_CHARS = 30


def split_sentences(text: str) -> list[str]:
    """Split page text into sentences, dropping whitespace-only fragments."""
    return [s.strip() for s in _SEGMENTER.segment(text) if s.strip()]


def chunk_page(text: str, chunk_sentences: int,
               overlap_sentences: int) -> list[str]:
    """Chunk one page of text into overlapping sentence windows.

    Example with chunk_sentences=8, overlap_sentences=4 (step = 4):
        chunk 0 = sentences 0..7
        chunk 1 = sentences 4..11
        chunk 2 = sentences 8..15  ...
    A page with <= 8 sentences yields a single chunk.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    step = max(1, chunk_sentences - overlap_sentences)
    chunks: list[str] = []
    for start in range(0, len(sentences), step):
        window = sentences[start:start + chunk_sentences]
        chunk = " ".join(window)
        if len(chunk) >= _MIN_CHUNK_CHARS:
            chunks.append(chunk)
        # Stop once a window reached the end of the page — any further
        # window would be a pure subset of this one.
        if start + chunk_sentences >= len(sentences):
            break
    return chunks
