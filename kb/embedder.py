"""Local embedding model (sentence-transformers).

The model runs entirely on this machine — downloaded once from Hugging Face
(~470 MB for the default multilingual MiniLM), then cached and used offline.
No tokens, no API costs.

Both the indexer (embedding chunks) and the search service (embedding
queries) use this module, guaranteeing the SAME model produces both sides
— vectors from different models are not comparable.
"""

from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None
_model_name: str | None = None


def get_model(model_name: str) -> SentenceTransformer:
    """Load the model once and reuse it (loading takes a few seconds)."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        # device=None lets sentence-transformers auto-pick the best device:
        # Apple Silicon GPU ("mps") when available, else CPU.
        _model = SentenceTransformer(model_name, device=None)
        _model_name = model_name
    return _model


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunk texts. Returns one vector per text.

    normalize_embeddings=True scales vectors to unit length so cosine
    similarity in Chroma behaves consistently.
    """
    model = get_model(model_name)
    vectors = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(model_name: str, query: str) -> list[float]:
    """Embed a single search query."""
    return embed_texts(model_name, [query])[0]
