"""Hybrid search: BM25 keyword + semantic vectors, merged with RRF.

Query flow (see also plan.txt section 4.2):

    user query ──┬── FTS5 MATCH (BM25)      → ranked list A  (keyword)
                 └── embed → Chroma (HNSW)  → ranked list B  (semantic)
    RRF: score(chunk) = Σ 1 / (rrf_k + rank_in_list)   over lists A and B
    → final top_k, best first

Reciprocal Rank Fusion uses only RANK POSITIONS, never raw scores —
BM25 scores and cosine similarities live on incompatible scales, so
normalizing them is fragile; ranks are always comparable. A chunk ranked
well in BOTH lists beats a chunk that tops only one, which is almost
always the result the user wanted.
"""

import html
import re
import sqlite3

from kb import database, embedder, vectors
from kb.config import Config

# Words in a result snippet when the hit came only from the semantic engine
# (FTS5 provides highlighted snippets for its own hits; for vector-only hits
# we build one from the start of the chunk).
_FALLBACK_SNIPPET_WORDS = 60


def _fallback_snippet(text: str, query: str) -> str:
    """First N words of the chunk, with any query words <mark>-highlighted."""
    words = text.split()
    snippet = " ".join(words[:_FALLBACK_SNIPPET_WORDS])
    if len(words) > _FALLBACK_SNIPPET_WORDS:
        snippet += " …"
    snippet = html.escape(snippet)
    # Highlight query words (case-insensitive, whole words) for display parity
    # with FTS5 snippets.
    for word in set(re.findall(r"\w+", query)):
        if len(word) < 3:
            continue  # skip tiny words: highlighting "of" everywhere is noise
        snippet = re.sub(
            rf"\b({re.escape(word)})\b", r"<mark>\1</mark>",
            snippet, flags=re.IGNORECASE,
        )
    return snippet


def hybrid_search(conn: sqlite3.Connection, cfg: Config, query: str,
                  top_k: int | None = None) -> list[dict]:
    """Run both engines and return the RRF-fused top results.

    Each result dict has: chunk_id, file_path, filename, page, snippet,
    rrf_score, bm25_rank, vector_rank (ranks are None if the chunk didn't
    appear in that engine's list — useful for debugging result quality).
    """
    top_k = top_k or cfg.top_k
    n = cfg.candidates_per_engine

    # --- Engine 1: keyword (BM25 via FTS5) ---------------------------------
    keyword_hits = database.keyword_search(conn, query, n)
    bm25_rank = {h["chunk_id"]: i + 1 for i, h in enumerate(keyword_hits)}
    # FTS5 marks hits with \x01/\x02 sentinels (see database.keyword_search):
    # escape the text for HTML safety first, then turn sentinels into tags.
    fts_snippets = {
        h["chunk_id"]: html.escape(h["snip"])
                           .replace("\x01", "<mark>")
                           .replace("\x02", "</mark>")
        for h in keyword_hits
    }

    # --- Engine 2: semantic (Chroma) ----------------------------------------
    query_vector = embedder.embed_query(cfg.model, query)
    semantic_ids = vectors.semantic_search(query_vector, n)
    vector_rank = {cid: i + 1 for i, cid in enumerate(semantic_ids)}

    # --- Fuse with Reciprocal Rank Fusion -----------------------------------
    rrf: dict[str, float] = {}
    for ranks in (bm25_rank, vector_rank):
        for chunk_id, rank in ranks.items():
            rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (cfg.rrf_k + rank)

    best_ids = sorted(rrf, key=rrf.get, reverse=True)[:top_k]

    # --- Build display results ----------------------------------------------
    results = []
    for chunk_id in best_ids:
        chunk = database.get_chunk(conn, chunk_id)
        if chunk is None:
            continue  # index halves briefly out of sync mid-reindex; skip
        file_path = chunk["file_path"]
        results.append({
            "chunk_id": chunk_id,
            "file_path": file_path,
            "filename": file_path.rsplit("/", 1)[-1],
            "page": chunk["page"],
            "snippet": fts_snippets.get(chunk_id)
                       or _fallback_snippet(chunk["text"], query),
            "rrf_score": round(rrf[chunk_id], 5),
            "bm25_rank": bm25_rank.get(chunk_id),
            "vector_rank": vector_rank.get(chunk_id),
        })
    return results
