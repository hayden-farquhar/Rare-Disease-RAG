"""FAISS vector store for chunk retrieval.

Manages FAISS indices with associated chunk metadata.
Uses IndexFlatIP for small collections (<100K), IndexIVFFlat for larger ones.
"""

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import faiss
import numpy as np

from src.index.chunker import Chunk

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "indices"


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore:
    """FAISS-backed vector store with metadata."""

    def __init__(self, name: str, dim: int = 384):
        self.name = name
        self.dim = dim
        self.index: Optional[faiss.Index] = None
        self.chunks: list[Chunk] = []

    def build_index(self, embeddings: np.ndarray, chunks: list[Chunk],
                    use_ivf: bool = False, nlist: int = 100):
        """Build FAISS index from embeddings and chunks.

        Args:
            embeddings: (n, dim) float32 array of normalised embeddings.
            chunks: Corresponding chunk objects.
            use_ivf: Use IVFFlat for large collections.
            nlist: Number of IVF clusters (only if use_ivf=True).
        """
        assert len(embeddings) == len(chunks), "Embeddings and chunks must match"
        assert embeddings.shape[1] == self.dim, f"Expected dim={self.dim}, got {embeddings.shape[1]}"

        n = len(embeddings)
        self.chunks = chunks

        if use_ivf and n > 1000:
            # IVF index for larger collections
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, min(nlist, n // 10),
                                            faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, nlist)
            print(f"Built IVF index '{self.name}': {n} vectors, {self.dim}d, nlist={min(nlist, n // 10)}")
        else:
            # Flat index for small collections (exact search)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(embeddings)
            print(f"Built flat index '{self.name}': {n} vectors, {self.dim}d")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Search the index for nearest neighbours."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self, directory: Optional[Path] = None):
        """Save index and metadata to disk."""
        directory = directory or INDEX_DIR / self.name
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(directory / "index.faiss"))

        # Save chunk metadata
        chunk_data = [c.to_dict() for c in self.chunks]
        with open(directory / "chunks.json", "w") as f:
            json.dump(chunk_data, f, indent=2)

        # Save config
        config = {"name": self.name, "dim": self.dim, "n_vectors": self.index.ntotal}
        with open(directory / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved index '{self.name}' to {directory}")

    @classmethod
    def load(cls, name: str, directory: Optional[Path] = None) -> "VectorStore":
        """Load index and metadata from disk."""
        directory = directory or INDEX_DIR / name
        if not directory.exists():
            raise FileNotFoundError(f"No index at {directory}")

        with open(directory / "config.json") as f:
            config = json.load(f)

        store = cls(name=config["name"], dim=config["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "chunks.json") as f:
            chunk_data = json.load(f)
        store.chunks = [Chunk.from_dict(d) for d in chunk_data]

        print(f"Loaded index '{name}': {store.index.ntotal} vectors, {store.dim}d")
        return store
