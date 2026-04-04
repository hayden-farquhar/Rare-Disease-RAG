"""Build FAISS index from PubMed case reports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.pubmed_cases import load_cases
from src.index.pubmed_chunker import chunk_all_cases
from src.index.embedder import get_embedder, embed_chunks
from src.index.vector_store import VectorStore


def main(embedding_model: str = "all-MiniLM-L6-v2"):
    print("=" * 60)
    print("Building PubMed Case Reports FAISS index")
    print("=" * 60)

    print("\n1. Loading case reports...")
    cases = load_cases()
    print(f"   Loaded {len(cases)} case reports")

    print("\n2. Creating chunks...")
    chunks = chunk_all_cases(cases)
    print(f"   Created {len(chunks)} chunks")

    if not chunks:
        print("No chunks to index!")
        return

    print(f"\n3. Embedding with {embedding_model}...")
    embedder = get_embedder(embedding_model)
    embeddings = embed_chunks(chunks, embedder)
    print(f"   Embeddings shape: {embeddings.shape}")

    print("\n4. Building FAISS index...")
    store = VectorStore(name="pubmed", dim=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save()

    print("\n" + "=" * 60)
    print("PubMed index built successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
