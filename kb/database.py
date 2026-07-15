"""SQLite layer: file manifest, FTS5 keyword index, and indexer run history.

One database file (data/kb.db) holds three things:

  manifest    — one row per PDF the crawler has seen. Used for change
                detection (mtime/size/hash) and to know what to purge when
                a file is modified or deleted.

  chunks_fts  — an FTS5 virtual table holding the raw text of every chunk.
                FTS5 maintains an inverted index and ranks matches with BM25.
                This is the entire "keyword search" side of the system.

  runs        — one row per indexer run, so the UI can show "last indexed at
                X, N files, M chunks".

Everything here is inspectable with any SQLite browser — handy for debugging
("what chunks does file X have?", "what does BM25 return for query Y?").
"""

import re
import sqlite3
from datetime import datetime

from kb.config import DB_PATH

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    path         TEXT PRIMARY KEY,  -- absolute file path
    mtime        REAL,              -- file modified time at last index
    size         INTEGER,           -- file size in bytes at last index
    content_hash TEXT,              -- sha1 of file contents at last index
    status       TEXT,              -- indexed | skipped_no_text |
                                    -- skipped_encrypted | failed
    page_count   INTEGER,
    chunk_count  INTEGER,
    last_indexed TEXT,              -- ISO timestamp
    error        TEXT               -- exception message when status=failed
);

-- FTS5 keyword index. Only `text` is searchable; the other columns are
-- stored payload (UNINDEXED) so a hit can be mapped back to its chunk.
-- Tokenizer: porter stemming ("filing" matches "filings") on top of
-- unicode61 (lowercasing, diacritics stripped). No stopword removal —
-- BM25's IDF weighting makes common words score ~0 automatically.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id  UNINDEXED,
    file_path UNINDEXED,
    page      UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started         TEXT,
    finished        TEXT,
    files_scanned   INTEGER,
    files_indexed   INTEGER,
    files_skipped   INTEGER,   -- no text layer / encrypted
    files_failed    INTEGER,
    files_deleted   INTEGER,   -- purged because the PDF disappeared
    chunks_added    INTEGER
);
"""


def connect() -> sqlite3.Connection:
    """Open (and if needed create) the database. Safe to call from anywhere."""
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# Manifest helpers (used by the indexer)
# ---------------------------------------------------------------------------

def get_manifest(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM manifest WHERE path = ?", (path,)).fetchone()


def upsert_manifest(conn: sqlite3.Connection, path: str, *, mtime: float,
                    size: int, content_hash: str, status: str,
                    page_count: int = 0, chunk_count: int = 0,
                    error: str = "") -> None:
    conn.execute(
        """INSERT INTO manifest
               (path, mtime, size, content_hash, status, page_count,
                chunk_count, last_indexed, error)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
               mtime=excluded.mtime, size=excluded.size,
               content_hash=excluded.content_hash, status=excluded.status,
               page_count=excluded.page_count, chunk_count=excluded.chunk_count,
               last_indexed=excluded.last_indexed, error=excluded.error""",
        (path, mtime, size, content_hash, status, page_count, chunk_count,
         datetime.now().isoformat(timespec="seconds"), error),
    )


def delete_manifest(conn: sqlite3.Connection, path: str) -> None:
    conn.execute("DELETE FROM manifest WHERE path = ?", (path,))


def all_manifest_paths(conn: sqlite3.Connection) -> list[str]:
    return [r["path"] for r in conn.execute("SELECT path FROM manifest")]


# ---------------------------------------------------------------------------
# Chunk (FTS5) helpers
# ---------------------------------------------------------------------------

def delete_file_chunks(conn: sqlite3.Connection, path: str) -> None:
    """Remove every chunk of a file from the keyword index.

    Called whenever a file changed (before re-inserting) or was deleted.
    The matching Chroma deletion happens in kb/vectors.py.
    """
    conn.execute("DELETE FROM chunks_fts WHERE file_path = ?", (path,))


def insert_chunks(conn: sqlite3.Connection,
                  rows: list[tuple[str, str, str, int]]) -> None:
    """Insert chunks as (text, chunk_id, file_path, page) tuples."""
    conn.executemany(
        "INSERT INTO chunks_fts (text, chunk_id, file_path, page) VALUES (?, ?, ?, ?)",
        rows,
    )


def get_chunk(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    """Fetch one chunk's text + metadata by id (used to build result snippets)."""
    return conn.execute(
        "SELECT text, chunk_id, file_path, page FROM chunks_fts WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()


# ---------------------------------------------------------------------------
# Keyword search (BM25)
# ---------------------------------------------------------------------------

def _build_match_expression(query: str) -> str:
    """Turn a free-text user query into a safe FTS5 MATCH expression.

    - Each word is double-quoted, which neutralizes FTS5 operator syntax
      (AND/OR/NEAR/*/etc.) so arbitrary user input can't break the query.
    - Words are joined with OR: BM25 naturally ranks chunks matching more
      (and rarer) words higher, without requiring every word to be present.
    - Runs of consecutive Capitalized words ("Rajesh Ramani", "Mavic Pro")
      are ALSO added as phrase terms, so chunks containing the words
      adjacent and in order get an extra scoring boost.
    """
    words = re.findall(r"\w+", query, re.UNICODE)
    if not words:
        return ""
    terms = [f'"{w}"' for w in words]

    # Detect capitalized runs in the original query for phrase boosting.
    tokens = query.split()
    run: list[str] = []
    for tok in tokens + [""]:  # trailing "" flushes the last run
        word = re.sub(r"\W+", "", tok)
        if word and word[0].isupper():
            run.append(word)
        else:
            if len(run) >= 2:
                terms.append('"' + " ".join(run) + '"')
            run = []
    return " OR ".join(terms)


def keyword_search(conn: sqlite3.Connection, query: str,
                   limit: int) -> list[dict]:
    """BM25-ranked keyword search. Returns hits best-first.

    Note: FTS5's bm25() returns NEGATIVE scores where lower = better,
    so a plain ascending ORDER BY puts the best match first.
    """
    match = _build_match_expression(query)
    if not match:
        return []
    # Highlight markers are control characters, not '<mark>' tags: the raw
    # PDF text may itself contain HTML characters, so kb/search.py first
    # HTML-escapes the snippet and then converts the markers to real tags.
    rows = conn.execute(
        """SELECT chunk_id, file_path, page,
                  snippet(chunks_fts, 0, char(1), char(2), ' … ', 20) AS snip,
                  bm25(chunks_fts) AS score
           FROM chunks_fts
           WHERE chunks_fts MATCH ?
           ORDER BY score
           LIMIT ?""",
        (match, limit),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Run history + stats (used by /status and the indexer)
# ---------------------------------------------------------------------------

def record_run(conn: sqlite3.Connection, summary: dict) -> None:
    conn.execute(
        """INSERT INTO runs (started, finished, files_scanned, files_indexed,
                             files_skipped, files_failed, files_deleted,
                             chunks_added)
           VALUES (:started, :finished, :files_scanned, :files_indexed,
                   :files_skipped, :files_failed, :files_deleted,
                   :chunks_added)""",
        summary,
    )


def last_run(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


def totals(conn: sqlite3.Connection) -> dict:
    files = conn.execute(
        "SELECT COUNT(*) FROM manifest WHERE status = 'indexed'"
    ).fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    return {"files_indexed": files, "total_chunks": chunks}
