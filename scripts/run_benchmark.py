"""Run retrieval benchmark on all curated cases (NEJM + test cases)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.evaluate.metrics import disease_match, top_k_accuracy, mean_reciprocal_rank


def main():
    print("=" * 70)
    print("Retrieval Benchmark")
    print("=" * 70)

    # Load index
    store = VectorStore.load("orphanet")
    embedder = get_embedder("all-MiniLM-L6-v2")

    # Load all benchmark cases
    benchmark_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json"]:
        fpath = benchmark_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                cases = json.load(f)
            all_cases.extend(cases)
            print(f"  Loaded {len(cases)} cases from {fname}")

    print(f"\nTotal benchmark cases: {len(all_cases)}")

    # Run retrieval
    all_predictions = []
    all_ground_truths = []
    all_aliases = []
    results = []

    for case in all_cases:
        vignette = case["clinical_vignette"]
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        case_id = case.get("case_id", case.get("title", "?"))

        # Embed and search
        query_emb = embedder.embed_query(vignette)
        hits = store.search(query_emb, top_k=20)

        # Extract unique disease names in order
        predictions = []
        seen = set()
        for hit in hits:
            if hit.chunk.disease_name not in seen:
                seen.add(hit.chunk.disease_name)
                predictions.append(hit.chunk.disease_name)

        # Find rank of correct diagnosis
        rank = None
        for i, pred in enumerate(predictions):
            if disease_match(pred, target, aliases):
                rank = i + 1
                break

        results.append({
            "case_id": case_id,
            "target": target,
            "rank": rank,
            "top3": predictions[:3],
        })
        all_predictions.append(predictions)
        all_ground_truths.append(target)
        all_aliases.append(aliases)

        rank_str = str(rank) if rank else "NOT FOUND"
        mark = "OK" if rank and rank <= 5 else "MISS" if rank is None else f"rank {rank}"
        print(f"  {case_id:<20} -> rank {rank_str:<12} [{mark}]  ({target})")

    # Compute metrics
    print(f"\n{'='*70}")
    print("METRICS")
    print(f"{'='*70}")

    for k in [1, 3, 5, 10]:
        acc = top_k_accuracy(all_predictions, all_ground_truths, all_aliases, k=k)
        print(f"  Top-{k} accuracy: {acc:.1%} ({int(acc * len(all_cases))}/{len(all_cases)})")

    mrr = mean_reciprocal_rank(all_predictions, all_ground_truths, all_aliases)
    print(f"  MRR: {mrr:.3f}")

    found = sum(1 for r in results if r["rank"] is not None)
    print(f"  Found in top-20: {found}/{len(results)} ({found/len(results):.1%})")

    # Save
    out_path = benchmark_dir / "retrieval_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "metrics": {
            "top1": top_k_accuracy(all_predictions, all_ground_truths, all_aliases, k=1),
            "top3": top_k_accuracy(all_predictions, all_ground_truths, all_aliases, k=3),
            "top5": top_k_accuracy(all_predictions, all_ground_truths, all_aliases, k=5),
            "top10": top_k_accuracy(all_predictions, all_ground_truths, all_aliases, k=10),
            "mrr": mrr,
        }}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
