"""Build all FAISS + BM25 indices (Orphanet + PubMed + targeted cases)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.orphanet_ingest import load_diseases
from src.ingest.pubmed_cases import load_cases, CaseReport
from src.index.chunker import chunk_all_diseases, Chunk
from src.index.pubmed_chunker import chunk_all_cases
from src.index.narrative_generator import load_narratives
from src.index.embedder import get_embedder, embed_chunks
from src.index.vector_store import VectorStore
from src.retrieve.bm25_store import BM25Store


def main(embedding_model: str = "all-MiniLM-L6-v2"):
    project_dir = Path(__file__).resolve().parents[1]

    print("=" * 60)
    print("Building ALL indices (Orphanet + PubMed)")
    print(f"  Embedding model: {embedding_model}")
    print("=" * 60)

    embedder = get_embedder(embedding_model)

    # === ORPHANET INDEX ===
    print("\n--- Orphanet Index ---")
    diseases = load_diseases()
    narratives = load_narratives()
    print(f"  {len(diseases)} diseases, {len(narratives)} narratives")

    chunks = chunk_all_diseases(diseases, narratives=narratives)
    print(f"  {len(chunks)} chunks")

    embeddings = embed_chunks(chunks, embedder)
    store = VectorStore(name="orphanet", dim=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save()

    bm25 = BM25Store(name="orphanet")
    bm25.build(chunks)
    bm25.save()

    # === PUBMED INDEX ===
    print("\n--- PubMed Index ---")
    pubmed_chunks = []

    # Original case reports
    try:
        cases = load_cases()
        from src.index.pubmed_chunker import chunk_all_cases
        pubmed_chunks.extend(chunk_all_cases(cases))
        print(f"  {len(cases)} original case reports -> {len(pubmed_chunks)} chunks")
    except FileNotFoundError:
        print("  No original case reports found")

    # Targeted case reports for benchmark diseases
    targeted_path = project_dir / "data" / "raw" / "pubmed_cases" / "benchmark_disease_cases.json"
    if targeted_path.exists():
        targeted = json.load(open(targeted_path))
        for tc in targeted:
            # Create a chunk from the case report text
            pubmed_chunks.append(Chunk(
                chunk_id=f"PMC:{tc['pmc_id']}_case",
                text=f"Case report: {tc['disease_name']}\n\n{tc['text'][:2000]}",
                source="pubmed",
                source_id=f"PMC:{tc['pmc_id']}",
                disease_name=tc['disease_name'],
                chunk_type="case_presentation",
                metadata={"pmc_id": tc['pmc_id'], "orpha_code": tc['orpha_code']},
            ))
        print(f"  {len(targeted)} targeted case reports added")

    if pubmed_chunks:
        print(f"  Total PubMed chunks: {len(pubmed_chunks)}")
        pub_embeddings = embed_chunks(pubmed_chunks, embedder)
        pub_store = VectorStore(name="pubmed", dim=pub_embeddings.shape[1])
        pub_store.build_index(pub_embeddings, pubmed_chunks)
        pub_store.save()

        pub_bm25 = BM25Store(name="pubmed")
        pub_bm25.build(pubmed_chunks)
        pub_bm25.save()

    print("\n" + "=" * 60)
    print("All indices built!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    main(args.embedding_model)
