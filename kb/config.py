"""Configuration loading and project paths.

Everything configurable lives in config.yaml at the project root.
This module loads it once and exposes it as a simple Config object,
plus the standard locations for generated data (data/ folder).
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Project root = the folder containing config.yaml (parent of this kb/ package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All generated artifacts live under data/ (gitignored).
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "kb.db"          # SQLite: manifest + FTS5 keyword index
CHROMA_DIR = DATA_DIR / "chroma"      # Chroma persistent vector store
LOG_PATH = DATA_DIR / "index.log"     # indexer log
LOCK_PATH = DATA_DIR / "indexer.lock" # prevents two indexers running at once


@dataclass
class Config:
    """Typed view of config.yaml. See that file for per-field documentation."""

    roots: list[Path] = field(default_factory=list)
    exclude_globs: list[str] = field(default_factory=list)
    model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_sentences: int = 8
    chunk_overlap_sentences: int = 4
    top_k: int = 10
    candidates_per_engine: int = 20
    rrf_k: int = 60
    port: int = 8130


def load_config() -> Config:
    """Read config.yaml and return a Config with expanded, absolute root paths."""
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
    cfg = Config(
        roots=[Path(r).expanduser().resolve() for r in raw.get("roots", [])],
        exclude_globs=raw.get("exclude_globs") or [],
        model=raw.get("model", Config.model),
        chunk_sentences=int(raw.get("chunk_sentences", Config.chunk_sentences)),
        chunk_overlap_sentences=int(
            raw.get("chunk_overlap_sentences", Config.chunk_overlap_sentences)
        ),
        top_k=int(raw.get("top_k", Config.top_k)),
        candidates_per_engine=int(
            raw.get("candidates_per_engine", Config.candidates_per_engine)
        ),
        rrf_k=int(raw.get("rrf_k", Config.rrf_k)),
        port=int(raw.get("port", Config.port)),
    )
    DATA_DIR.mkdir(exist_ok=True)
    return cfg
