"""Multi-source retrieval with reciprocal rank fusion, BM25 hybrid, and reranking."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.index.vector_store import VectorStore, SearchResult
from src.index.embedder import SentenceTransformerEmbedder


@dataclass
class RetrievedChunk:
    """A retrieved chunk with fusion score and source attribution."""
    chunk_id: str
    text: str
    source: str
    source_id: str
    disease_name: str
    chunk_type: str
    score: float
    hpo_terms: list[str] = field(default_factory=list)


class MultiSourceRetriever:
    """Retrieve from multiple knowledge bases and fuse results.

    Supports dense (FAISS) + sparse (BM25/TF-IDF) hybrid retrieval
    with reciprocal rank fusion.
    """

    def __init__(
        self,
        stores: dict[str, VectorStore],
        embedder: Optional[SentenceTransformerEmbedder] = None,
        source_weights: Optional[dict[str, float]] = None,
        bm25_stores: Optional[dict] = None,
        bm25_weight: float = 0.8,
        phenotype_scorer=None,
        phenotype_weight: float = 1.0,
        disease_chunks_map: Optional[dict] = None,
    ):
        self.stores = stores
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.source_weights = source_weights or {
            "orphanet": 1.2,
            "omim": 1.1,
            "pubmed": 1.0,
        }
        self.bm25_stores = bm25_stores or {}
        self.bm25_weight = bm25_weight
        self.phenotype_scorer = phenotype_scorer
        self.phenotype_weight = phenotype_weight
        self.disease_chunks_map = disease_chunks_map or {}

    def retrieve(self, queries: list[str], top_k: int = 20,
                 per_query_k: int = 20,
                 query_hpo_ids: Optional[list[str]] = None,
                 hyde_document: Optional[str] = None) -> list[RetrievedChunk]:
        """Retrieve across all sources and queries, fuse with RRF."""
        all_rankings: list[tuple[str, list]] = []  # (label, ranked results)

        # Dense retrieval per source per query
        for source_name, store in self.stores.items():
            source_results = []
            for query in queries:
                q_emb = self.embedder.embed_query(query)
                hits = store.search(q_emb, top_k=per_query_k)
                source_results.extend(hits)
            weight = self.source_weights.get(source_name, 1.0)
            all_rankings.append((f"dense_{source_name}", source_results, weight))

        # BM25 retrieval per source per query
        for source_name, bm25_store in self.bm25_stores.items():
            source_results = []
            for query in queries:
                hits = bm25_store.search(query, top_k=per_query_k)
                source_results.extend(hits)
            all_rankings.append((f"bm25_{source_name}", source_results, self.bm25_weight))

        # HyDE retrieval: embed hypothetical document as a query
        if hyde_document:
            hyde_emb = self.embedder.embed_query(hyde_document)
            for source_name, store in self.stores.items():
                hits = store.search(hyde_emb, top_k=per_query_k)
                weight = self.source_weights.get(source_name, 1.0) * 1.5  # boost HyDE
                all_rankings.append((f"hyde_{source_name}", hits, weight))

        # HPO phenotype-based retrieval (if scorer and HPO terms provided)
        if self.phenotype_scorer and query_hpo_ids and self.disease_chunks_map:
            ranked_diseases = self.phenotype_scorer.rank_diseases(query_hpo_ids, top_k=per_query_k)
            pheno_results = []
            for orpha_code, disease_name, score in ranked_diseases:
                chunks = self.disease_chunks_map.get(disease_name, [])
                for chunk in chunks[:2]:  # Max 2 chunks per disease
                    from src.retrieve.bm25_store import BM25Result
                    pheno_results.append(BM25Result(chunk=chunk, score=score))
            if pheno_results:
                all_rankings.append(("hpo_phenotype", pheno_results, self.phenotype_weight))

        # Reciprocal Rank Fusion across all ranking lists
        fused = self._reciprocal_rank_fusion(all_rankings)

        return fused[:top_k]

    def _reciprocal_rank_fusion(
        self, rankings: list[tuple], k: int = 60
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion across multiple ranking lists.

        Each ranking is (label, results_list, weight).
        RRF score = sum over rankings of: weight / (k + rank)
        """
        chunk_scores: dict[str, float] = {}
        chunk_data: dict[str, dict] = {}  # chunk_id -> chunk info

        for label, results, weight in rankings:
            # Sort by score descending to get rankings
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

            # Deduplicate within this ranking list
            seen = set()
            rank = 0
            for result in sorted_results:
                # Handle both SearchResult and BM25Result
                chunk = result.chunk
                cid = chunk.chunk_id
                if cid in seen:
                    continue
                seen.add(cid)
                rank += 1

                rrf_score = weight / (k + rank)
                chunk_scores[cid] = chunk_scores.get(cid, 0.0) + rrf_score

                # Store chunk data (keep first seen version)
                if cid not in chunk_data:
                    chunk_data[cid] = {
                        "text": chunk.text,
                        "source": chunk.source,
                        "source_id": chunk.source_id,
                        "disease_name": chunk.disease_name,
                        "chunk_type": chunk.chunk_type,
                        "hpo_terms": chunk.hpo_terms,
                    }

        # Sort by fused score
        ranked_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)

        results = []
        for cid in ranked_ids:
            cd = chunk_data[cid]
            results.append(RetrievedChunk(
                chunk_id=cid,
                text=cd["text"],
                source=cd["source"],
                source_id=cd["source_id"],
                disease_name=cd["disease_name"],
                chunk_type=cd["chunk_type"],
                score=chunk_scores[cid],
                hpo_terms=cd["hpo_terms"],
            ))

        return results


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk],
               top_k: int = 10) -> list[RetrievedChunk]:
        """Rerank chunks by cross-encoder relevance score."""
        if not chunks:
            return []

        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Update scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda c: c.score, reverse=True)
        return reranked[:top_k]
