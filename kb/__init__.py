# kb — the shared library used by both indexer.py and server.py.
#
# Modules:
#   config.py    — loads config.yaml, defines all project paths
#   database.py  — SQLite: file manifest, FTS5 keyword index (BM25), run history
#   chunker.py   — sentence-aligned chunking with overlap
#   embedder.py  — local sentence-transformers embedding model
#   vectors.py   — Chroma vector store (semantic index, HNSW)
#   search.py    — hybrid search: BM25 + vectors merged with RRF
