"""Test basic retrieval against known rare diseases.

Usage:
    python scripts/test_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.evaluate.metrics import disease_match


def main():
    print("=" * 60)
    print("Testing retrieval pipeline")
    print("=" * 60)

    # Load index
    print("\n1. Loading index...")
    store = VectorStore.load("orphanet")
    embedder = get_embedder("all-MiniLM-L6-v2")

    # Load test cases
    print("\n2. Loading test cases...")
    test_path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "test_cases.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    # Run retrieval for each test case
    print(f"\n3. Testing retrieval on {len(test_cases)} cases...\n")

    for case in test_cases:
        print(f"--- Case: {case['title']} ---")
        print(f"Target: {case['final_diagnosis']} (ORPHA:{case.get('orpha_code', '?')})")

        # Embed the vignette
        query_emb = embedder.embed_query(case["clinical_vignette"])

        # Search
        results = store.search(query_emb, top_k=10)

        # Check if target disease appears in results
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        found_rank = None

        print(f"Top 10 results:")
        for i, result in enumerate(results):
            disease = result.chunk.disease_name
            score = result.score
            match = ""
            if disease_match(disease, target, aliases):
                if found_rank is None:
                    found_rank = i + 1
                match = " <<< MATCH"
            print(f"  {i+1}. [{score:.4f}] {disease} ({result.chunk.chunk_type}){match}")

        if found_rank:
            print(f"  => Target found at rank {found_rank}")
        else:
            print(f"  => Target NOT found in top 10")
        print()

    print("=" * 60)
    print("Retrieval test complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
