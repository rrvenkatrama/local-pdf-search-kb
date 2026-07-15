"""Chroma vector store — the semantic half of the search system.

Chroma runs embedded (inside this process, no server) and persists to
data/chroma/. Internally it maintains an HNSW index over the vectors,
which is what makes nearest-neighbor lookup fast; we never manage HNSW
directly.

Chunk IDs mirror the ones in the FTS5 table, so results from both engines
can be merged by id (see kb/search.py).
"""

import chromadb

from kb.config import CHROMA_DIR

_COLLECTION = "pdf_chunks"

_client: chromadb.ClientAPI | None = None


def _collection():
    """Open the persistent collection (created on first use)."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # cosine distance — pairs with the normalized vectors from kb/embedder.py
    return _client.get_or_create_collection(
        _COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def refresh() -> None:
    """Drop the cached client so the next call re-reads the store from disk.

    Needed because Chroma runs EMBEDDED: when the indexer (a separate
    process) writes new vectors, a long-running server process keeps
    serving its in-memory snapshot until the client is recreated.
    server.py calls this whenever it notices a new indexer run finished.
    """
    global _client
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    _client = None


# Chroma rejects add() calls above ~5,461 items; large PDFs can produce
# more chunks than that, so inserts are split into batches.
_MAX_ADD_BATCH = 5000


def add_chunks(ids: list[str], vectors: list[list[float]],
               texts: list[str], metadatas: list[dict]) -> None:
    """Store embedded chunks. metadatas carry file_path and page for each."""
    collection = _collection()
    for i in range(0, len(ids), _MAX_ADD_BATCH):
        j = i + _MAX_ADD_BATCH
        collection.add(ids=ids[i:j], embeddings=vectors[i:j],
                       documents=texts[i:j], metadatas=metadatas[i:j])


def delete_file_chunks(path: str) -> None:
    """Remove every chunk of a file (file modified or deleted).

    Chroma supports delete-by-metadata-filter, so we don't need to know
    the individual chunk ids.
    """
    _collection().delete(where={"file_path": path})


def semantic_search(query_vector: list[float], limit: int) -> list[str]:
    """Return chunk ids ranked by cosine similarity to the query vector."""
    result = _collection().query(
        query_embeddings=[query_vector],
        n_results=limit,
        include=[],  # ids are always returned; we fetch text from SQLite
    )
    return result["ids"][0] if result["ids"] else []


def count() -> int:
    return _collection().count()
