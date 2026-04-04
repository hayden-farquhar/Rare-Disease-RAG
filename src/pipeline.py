"""End-to-end RAG pipeline for rare disease diagnosis.

Orchestrates: query processing -> retrieval -> context assembly -> generation.
"""

import json
from pathlib import Path
from typing import Optional

from src.index.vector_store import VectorStore
from src.index.embedder import SentenceTransformerEmbedder, get_embedder
from src.index.chunker import Chunk, chunk_all_diseases
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.retrieve.retriever import MultiSourceRetriever, CrossEncoderReranker
from src.retrieve.query_processor import process_query
from src.retrieve.context_assembler import assemble_context
from src.generate.generator import generate_diagnosis, generate_diagnosis_no_rag, DiagnosticOutput

PROJECT_DIR = Path(__file__).resolve().parents[1]


class RareDiseaseRAG:
    """End-to-end RAG pipeline for rare disease diagnosis."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        generation_model: str = "claude-haiku-4-5-20251001",
        use_reranker: bool = True,
        use_llm_queries: bool = True,
    ):
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.use_reranker = use_reranker
        self.use_llm_queries = use_llm_queries

        self.embedder: Optional[SentenceTransformerEmbedder] = None
        self.stores: dict[str, VectorStore] = {}
        self.retriever: Optional[MultiSourceRetriever] = None
        self.reranker: Optional[CrossEncoderReranker] = None
        self.hpo: Optional[HPOOntology] = None
        self.anthropic_client = None

    def load(self):
        """Load all components from disk."""
        print("Loading RAG pipeline components...")

        # Embedder
        self.embedder = get_embedder(self.embedding_model)

        # Vector stores
        index_dir = PROJECT_DIR / "data" / "processed" / "indices"
        for source in ["orphanet", "omim", "pubmed"]:
            source_dir = index_dir / source
            if source_dir.exists():
                self.stores[source] = VectorStore.load(source, source_dir)

        if not self.stores:
            raise FileNotFoundError("No indices found. Run build_index() first.")

        # Retriever
        self.retriever = MultiSourceRetriever(
            stores=self.stores,
            embedder=self.embedder,
        )

        # Reranker
        if self.use_reranker:
            self.reranker = CrossEncoderReranker()

        # HPO ontology
        self.hpo = HPOOntology()
        hpo_index = PROJECT_DIR / "data" / "hpo" / "hpo_index.json"
        if hpo_index.exists():
            self.hpo.load_index(hpo_index)
        else:
            hpo_obo = PROJECT_DIR / "data" / "hpo" / "hp.obo"
            if hpo_obo.exists():
                self.hpo.load_obo(hpo_obo)
            else:
                print("Warning: HPO ontology not found. HPO mapping disabled.")
                self.hpo = None

        # Anthropic client
        import anthropic
        self.anthropic_client = anthropic.Anthropic()

        print(f"Pipeline loaded: {len(self.stores)} index(es), "
              f"reranker={'yes' if self.reranker else 'no'}, "
              f"HPO={'yes' if self.hpo else 'no'}")

    def diagnose(self, vignette: str, top_k_retrieval: int = 20,
                 top_k_rerank: int = 10) -> DiagnosticOutput:
        """Run the full RAG diagnostic pipeline on a clinical vignette."""
        # Step 1: Query processing
        processed = process_query(
            vignette,
            hpo_ontology=self.hpo,
            use_llm=self.use_llm_queries,
            client=self.anthropic_client,
        )
        print(f"Extracted {len(processed.extracted_phenotypes)} phenotypes, "
              f"generated {len(processed.retrieval_queries)} queries")

        # Step 2: Multi-source retrieval
        retrieved = self.retriever.retrieve(
            queries=processed.retrieval_queries,
            top_k=top_k_retrieval,
        )
        print(f"Retrieved {len(retrieved)} chunks")

        # Step 3: Reranking (optional)
        if self.reranker and retrieved:
            retrieved = self.reranker.rerank(
                query=vignette,
                chunks=retrieved,
                top_k=top_k_rerank,
            )
            print(f"Reranked to {len(retrieved)} chunks")

        # Step 4: Context assembly
        context = assemble_context(retrieved)

        # Step 5: Diagnostic generation
        output = generate_diagnosis(
            vignette=vignette,
            context=context,
            client=self.anthropic_client,
            model=self.generation_model,
        )

        return output

    def diagnose_no_rag(self, vignette: str) -> DiagnosticOutput:
        """Baseline: LLM-only diagnosis without RAG."""
        return generate_diagnosis_no_rag(
            vignette=vignette,
            client=self.anthropic_client,
            model=self.generation_model,
        )


def build_orphanet_index(
    max_diseases: int = 0,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> VectorStore:
    """Build the Orphanet FAISS index from fetched disease data."""
    from src.index.embedder import embed_chunks

    print("Loading Orphanet disease data...")
    diseases = load_diseases()
    if max_diseases > 0:
        diseases = diseases[:max_diseases]

    print(f"Chunking {len(diseases)} diseases...")
    chunks = chunk_all_diseases(diseases)
    print(f"Created {len(chunks)} chunks")

    embedder = get_embedder(embedding_model)
    embeddings = embed_chunks(chunks, embedder)

    store = VectorStore(name="orphanet", dim=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save()

    return store
