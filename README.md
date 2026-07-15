# Local PDF Search KB

Fully local hybrid search over your PDF documents — keyword (BM25) **and**
semantic (embeddings), fused with Reciprocal Rank Fusion. No cloud, no API
keys, no token costs. Search only: no LLM, no generation.

## How it works

```
              ┌─────────────────────────────────────────────┐
 launchd 8:00 │  indexer.py                                  │
 UI button ──▶│  crawl ~/Documents → extract → sentence-     │
              │  chunk → embed locally → write both indexes  │
              └───────────────┬─────────────────────────────┘
                              ▼
                    data/  ── chroma/  (vectors, HNSW — semantic)
                           ── kb.db    (FTS5/BM25 — keyword; + manifest)
                              ▲
              ┌───────────────┴─────────────────────────────┐
 browser ────▶│  server.py (FastAPI, port 8130)              │
              │  query → BM25 top-20 + vector top-20         │
              │        → RRF merge → top-10 w/ snippets      │
              └─────────────────────────────────────────────┘
```

- **Chunking:** complete sentences only (pysbd), 8-sentence windows with
  4-sentence overlap, never crossing page boundaries.
- **Change detection:** mtime+size fast path, sha1 to confirm; modified files
  have all old chunks deleted from both indexes before re-inserting; deleted
  files are purged.
- **Skipped with a log entry:** scanned PDFs (no text layer) and
  password-protected PDFs. See `data/index.log`.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` via
  sentence-transformers, in-process, downloaded once (~470 MB) then offline.

## Setup

```bash
./install.sh                    # venv + deps + daily 08:00 launchd schedule
./venv/bin/python indexer.py    # first full index (few minutes)
./venv/bin/python server.py     # search service → http://localhost:8130/
```

## Using it

Open <http://localhost:8130/>, type anything — a question, a name, a phrase.
Results are grouped by document with highlighted snippets and page numbers;
**Open** launches the PDF in Preview/Acrobat. The footer shows index status
and a **Re-index now** button.

## Files

| Path | What it is |
|---|---|
| `config.yaml` | folders to index, model, chunking, ports — all settings |
| `indexer.py` | the crawler (run daily by launchd + on demand) |
| `server.py` | FastAPI search service + static UI |
| `kb/` | shared library (see `kb/__init__.py` for a module map) |
| `static/index.html` | the single-file web UI |
| `launchd/`, `install.sh` | daily schedule installation |
| `data/` | generated: vector store, SQLite index, logs (gitignored) |
| `plan.txt` | full requirements and design decisions |

## Handy debugging

```bash
sqlite3 data/kb.db "SELECT status, COUNT(*) FROM manifest GROUP BY status"
sqlite3 data/kb.db "SELECT snippet(chunks_fts,0,'[',']','…',10)
                    FROM chunks_fts WHERE chunks_fts MATCH 'mavic' LIMIT 5"
tail -f data/index.log
```
