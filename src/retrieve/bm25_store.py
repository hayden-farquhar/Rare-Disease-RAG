"""BM25/TF-IDF sparse retrieval store for hybrid search.

Uses scikit-learn TfidfVectorizer for sparse keyword matching,
complementing dense vector retrieval for rare disease terminology.
"""

import json
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.index.chunker import Chunk

INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "indices"


@dataclass
class BM25Result:
    chunk: Chunk
    score: float


class BM25Store:
    """TF-IDF based sparse retrieval store."""

    def __init__(self, name: str):
        self.name = name
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]):
        """Build TF-IDF index from chunks."""
        self.chunks = chunks
        texts = [c.text for c in chunks]

        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            sublinear_tf=True,  # Use log(1+tf) — closer to BM25
            min_df=2,
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"Built BM25 index '{self.name}': {self.tfidf_matrix.shape[0]} docs, "
              f"{self.tfidf_matrix.shape[1]} features")

    def search(self, query: str, top_k: int = 20) -> list[BM25Result]:
        """Search using TF-IDF cosine similarity."""
        if self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(BM25Result(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                ))
        return results

    def save(self, directory: Optional[Path] = None):
        """Save TF-IDF model and index to disk."""
        directory = directory or INDEX_DIR / self.name
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "bm25_vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(directory / "bm25_matrix.pkl", "wb") as f:
            pickle.dump(self.tfidf_matrix, f)

        # Chunks are already saved by VectorStore, but save a reference
        config = {"name": self.name, "n_docs": len(self.chunks)}
        with open(directory / "bm25_config.json", "w") as f:
            json.dump(config, f)

        print(f"Saved BM25 index '{self.name}' to {directory}")

    @classmethod
    def load(cls, name: str, chunks: list[Chunk],
             directory: Optional[Path] = None) -> "BM25Store":
        """Load TF-IDF model from disk. Requires chunks to be passed in."""
        directory = directory or INDEX_DIR / name

        store = cls(name=name)
        store.chunks = chunks

        with open(directory / "bm25_vectorizer.pkl", "rb") as f:
            store.vectorizer = pickle.load(f)
        with open(directory / "bm25_matrix.pkl", "rb") as f:
            store.tfidf_matrix = pickle.load(f)

        print(f"Loaded BM25 index '{name}': {store.tfidf_matrix.shape[0]} docs")
        return store
