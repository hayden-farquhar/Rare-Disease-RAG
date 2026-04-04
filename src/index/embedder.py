"""Embedding generation for document chunks.

Supports sentence-transformers (local, free) and OpenAI embedding models.
Default: all-MiniLM-L6-v2 (384 dims) for development.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from src.index.chunker import Chunk


class SentenceTransformerEmbedder:
    """Generate embeddings using sentence-transformers (local, free)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dim}")

    def embed_texts(self, texts: list[str], batch_size: int = 32,
                    show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts. Returns (n, dim) float32 array."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # For cosine similarity via dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query. Returns (dim,) float32 array."""
        return self.embed_texts([query], show_progress=False)[0]


class OpenAIEmbedder:
    """Generate embeddings using OpenAI API (text-embedding-3-small)."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.dim = 1536 if "3-small" in model else 3072

    def embed_texts(self, texts: list[str], batch_size: int = 100,
                    show_progress: bool = True) -> np.ndarray:
        """Embed texts via OpenAI API."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.model)
            batch_embs = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embs)
            if show_progress:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query], show_progress=False)[0]


def get_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Factory to get the appropriate embedder."""
    if model_name.startswith("text-embedding"):
        return OpenAIEmbedder(model=model_name)
    return SentenceTransformerEmbedder(model_name=model_name)


def embed_chunks(chunks: list[Chunk], embedder=None) -> np.ndarray:
    """Embed all chunks, returning (n, dim) array."""
    if embedder is None:
        embedder = get_embedder()
    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    return embedder.embed_texts(texts)
