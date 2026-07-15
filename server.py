#!/usr/bin/env python3
"""FastAPI search service.

Serves the HTML UI and four JSON endpoints:

  POST /search   {query, top_k?}  → hybrid search results (see kb/search.py)
  POST /open     {path}           → open the PDF in Preview/Acrobat (macOS `open`)
  POST /reindex                   → launch indexer.py in the background
  GET  /status                    → last run summary + totals + indexing flag

Start with:  ./venv/bin/python server.py     (UI at http://localhost:8130/)

The embedding model loads once at startup (a few seconds), so the first
search is fast. SQLite connections are per-request — cheap, and it keeps
this file free of threading concerns.
"""

import subprocess
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kb import database, embedder, search, vectors
from kb.config import LOCK_PATH, PROJECT_ROOT, load_config

cfg = load_config()
app = FastAPI(title="Local PDF Search KB")

# Id of the newest completed indexer run this process has seen. When a new
# run appears (daily launchd job or the UI's Re-index button — both separate
# processes), the embedded Chroma client must be refreshed to see the new
# vectors. Checking is one indexed SELECT per search — effectively free.
_seen_run_id: int | None = None


def _refresh_vectors_if_new_run(conn) -> None:
    global _seen_run_id
    run = database.last_run(conn)
    run_id = run["id"] if run else None
    if run_id != _seen_run_id:
        if _seen_run_id is not None:  # skip the no-op refresh at startup
            vectors.refresh()
        _seen_run_id = run_id


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class OpenRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/search")
def search_endpoint(req: SearchRequest) -> dict:
    """Hybrid (BM25 + semantic + RRF) search over all indexed chunks."""
    query = req.query.strip()
    if not query:
        return {"query": query, "results": []}
    conn = database.connect()
    try:
        _refresh_vectors_if_new_run(conn)
        results = search.hybrid_search(conn, cfg, query, req.top_k)
    finally:
        conn.close()
    return {"query": query, "results": results}


@app.post("/open")
def open_endpoint(req: OpenRequest) -> dict:
    """Open a result PDF in the default macOS viewer (Preview/Acrobat).

    Safety: only paths that live under a configured root AND are present in
    the manifest can be opened — the endpoint can't be used to open
    arbitrary files.
    """
    path = Path(req.path).resolve()
    if not any(path.is_relative_to(root) for root in cfg.roots):
        raise HTTPException(403, "Path is outside the configured roots")
    conn = database.connect()
    try:
        known = database.get_manifest(conn, str(path))
    finally:
        conn.close()
    if known is None or not path.exists():
        raise HTTPException(404, "File not found in the index")
    subprocess.run(["open", str(path)], check=False)
    return {"opened": str(path)}


@app.post("/reindex")
def reindex_endpoint() -> dict:
    """Start the indexer in the background (same script launchd runs daily).

    The indexer's lock file prevents double-runs; output goes to
    data/index.log as usual.
    """
    if LOCK_PATH.exists():
        raise HTTPException(409, "An index run is already in progress")
    subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "indexer.py")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,   # indexer logs to data/index.log itself
        stderr=subprocess.DEVNULL,
        start_new_session=True,      # keeps running if the server restarts
    )
    return {"started": True}


@app.get("/status")
def status_endpoint() -> dict:
    """Everything the UI footer shows: totals, last run, whether indexing now."""
    conn = database.connect()
    try:
        return {
            "indexing": LOCK_PATH.exists(),
            "totals": database.totals(conn),
            "last_run": database.last_run(conn),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Static UI (registered last so API routes take precedence)
# ---------------------------------------------------------------------------

@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"))


if __name__ == "__main__":
    # Warm the embedding model before accepting requests.
    print(f"Loading embedding model ({cfg.model}) …")
    embedder.get_model(cfg.model)
    print(f"Ready — UI at http://localhost:{cfg.port}/")
    uvicorn.run(app, host="127.0.0.1", port=cfg.port, log_level="warning")
