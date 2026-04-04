"""Build FAISS index from fetched Orphanet data.

Usage:
    python scripts/build_index.py [--embedding-model MODEL] [--no-narratives]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.orphanet_ingest import load_diseases
from src.index.chunker import chunk_all_diseases
from src.index.narrative_generator import load_narratives
from src.index.embedder import get_embedder, embed_chunks
from src.index.vector_store import VectorStore
from src.retrieve.bm25_store import BM25Store


def main(embedding_model: str = "all-MiniLM-L6-v2", use_narratives: bool = True):
    print("=" * 60)
    print("Building Orphanet FAISS index")
    print(f"  Embedding model: {embedding_model}")
    print(f"  Narratives: {'ON' if use_narratives else 'OFF'}")
    print("=" * 60)

    # Load disease data
    print("\n1. Loading disease data...")
    diseases = load_diseases()
    print(f"   Loaded {len(diseases)} diseases")

    # Stats
    with_hpo = sum(1 for d in diseases if d.hpo_associations)
    with_genes = sum(1 for d in diseases if d.genes)
    total_hpo = sum(len(d.hpo_associations) for d in diseases)
    print(f"   {with_hpo} diseases have HPO associations ({total_hpo} total)")
    print(f"   {with_genes} diseases have gene associations")

    # Load narratives
    narratives = {}
    if use_narratives:
        narratives = load_narratives()
        print(f"   Loaded {len(narratives)} narrative descriptions")

    # Chunk
    print("\n2. Creating chunks...")
    chunks = chunk_all_diseases(diseases, narratives=narratives)
    print(f"   Created {len(chunks)} chunks")
    chunk_types = {}
    for c in chunks:
        chunk_types[c.chunk_type] = chunk_types.get(c.chunk_type, 0) + 1
    for ct, count in sorted(chunk_types.items()):
        print(f"   - {ct}: {count}")

    # Embed
    print(f"\n3. Embedding with {embedding_model}...")
    embedder = get_embedder(embedding_model)
    embeddings = embed_chunks(chunks, embedder)
    print(f"   Embeddings shape: {embeddings.shape}")

    # Build and save FAISS index
    print("\n4. Building FAISS index...")
    store = VectorStore(name="orphanet", dim=embeddings.shape[1])
    store.build_index(embeddings, chunks)
    store.save()

    # Build and save BM25 index
    print("\n5. Building BM25 index...")
    bm25 = BM25Store(name="orphanet")
    bm25.build(chunks)
    bm25.save()

    print("\n" + "=" * 60)
    print("Index built successfully!")
    print("=" * 60)

    return store


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--no-narratives", action="store_true")
    args = parser.parse_args()
    main(args.embedding_model, use_narratives=not args.no_narratives)
