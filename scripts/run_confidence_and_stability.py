"""Task 6: Confidence calibration + Task 7: Multi-run stability.

Runs the RAG pipeline (optimised: no reranker) multiple times, capturing
confidence labels and full differential outputs for analysis.

Run 1 = confidence calibration data
Runs 1-3 = stability assessment
"""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.retrieve.retriever import MultiSourceRetriever, CrossEncoderReranker
from src.retrieve.bm25_store import BM25Store
from src.retrieve.context_assembler import assemble_context
from src.retrieve.phenotype_scorer import PhenotypeScorer
from src.generate.generator import generate_diagnosis
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
CACHE_PATH = BENCHMARK_DIR / "query_cache.json"


def load_query_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def get_processed_query(vignette, hpo, client, cache):
    from src.retrieve.query_processor import ProcessedQuery
    key = vignette[:200]
    if key in cache:
        return ProcessedQuery(**cache[key])
    from src.retrieve.query_processor import process_query
    processed = process_query(vignette, hpo_ontology=hpo, use_llm=True, client=client)
    if processed.extracted_phenotypes:
        processed.retrieval_queries.append(
            "rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5])
        )
    cache[key] = processed.model_dump()
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    return processed


def run_single_pass(all_cases, retriever, hpo, client, query_cache, run_id):
    """Run RAG pipeline (no reranker) on all cases. Returns per-case results with confidence."""
    results = []
    for i, case in enumerate(all_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        rarity = case.get("rarity", "unknown")
        vignette = case["clinical_vignette"]

        # Query processing (cached)
        processed = get_processed_query(vignette, hpo, client, query_cache)
        queries = processed.retrieval_queries or [vignette[:500]]

        # HPO IDs
        query_hpo_ids = []
        if processed.hpo_candidates:
            query_hpo_ids = [h["hpo_id"] for h in processed.hpo_candidates if h.get("score", 0) > 0.5]

        # Retrieve (no reranker — optimised config)
        retrieved = retriever.retrieve(
            queries=queries, top_k=50, per_query_k=25,
            query_hpo_ids=query_hpo_ids,
        )

        # Context + generation
        context = assemble_context(retrieved, max_tokens=6000)
        output = generate_diagnosis(vignette=vignette, context=context, client=client)

        # Extract predictions with confidence
        preds = []
        for dx in output.differential_diagnosis:
            preds.append({
                "disease_name": dx.disease_name,
                "confidence": dx.confidence,
                "rank": dx.rank,
            })

        # Find rank of correct answer
        correct_rank = None
        for j, dx in enumerate(output.differential_diagnosis):
            if disease_match(dx.disease_name, target, aliases):
                correct_rank = j + 1
                break

        result = {
            "case_id": case_id,
            "target": target,
            "rarity": rarity,
            "correct_rank": correct_rank,
            "predictions": preds,
            "top1_confidence": preds[0]["confidence"] if preds else "none",
        }
        results.append(result)

        rank_str = str(correct_rank) if correct_rank else "MISS"
        conf = result["top1_confidence"]
        print(f"  Run {run_id} [{i+1}/{len(all_cases)}] {case_id}: rank={rank_str} conf={conf}")

        time.sleep(0.5)

    return results


def analyse_confidence(all_runs):
    """Task 6: Confidence calibration analysis."""
    print(f"\n{'='*80}")
    print("TASK 6: CONFIDENCE CALIBRATION")
    print(f"{'='*80}")

    # Use run 1 for confidence analysis
    run1 = all_runs[0]

    # Stratify by confidence level
    conf_levels = ["high", "medium", "low"]
    for subset_label, filter_fn in [("ALL", lambda r: True), ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare")]:
        print(f"\n--- {subset_label} ---")
        print(f"{'Confidence':<12} {'n':<6} {'Top-1 Acc':<12} {'Top-3 Acc':<12} {'Top-5 Acc':<12}")
        print("-" * 54)

        subset = [r for r in run1 if filter_fn(r)]

        for conf in conf_levels:
            cases = [r for r in subset if r["top1_confidence"] == conf]
            if not cases:
                print(f"{conf:<12} {0:<6} {'—':<12} {'—':<12} {'—':<12}")
                continue
            n = len(cases)
            t1 = sum(1 for r in cases if r["correct_rank"] is not None and r["correct_rank"] <= 1) / n
            t3 = sum(1 for r in cases if r["correct_rank"] is not None and r["correct_rank"] <= 3) / n
            t5 = sum(1 for r in cases if r["correct_rank"] is not None and r["correct_rank"] <= 5) / n
            print(f"{conf:<12} {n:<6} {t1:<12.1%} {t3:<12.1%} {t5:<12.1%}")

        # Also report "none" if present
        none_cases = [r for r in subset if r["top1_confidence"] not in conf_levels]
        if none_cases:
            n = len(none_cases)
            t1 = sum(1 for r in none_cases if r["correct_rank"] is not None and r["correct_rank"] <= 1) / n
            print(f"{'other':<12} {n:<6} {t1:<12.1%}")

    # Overall calibration summary
    print(f"\nCalibration check: Is high confidence more accurate than low?")
    ur = [r for r in run1 if r["rarity"] == "ultra-rare"]
    for conf in conf_levels:
        cases = [r for r in ur if r["top1_confidence"] == conf]
        if cases:
            acc = sum(1 for r in cases if r["correct_rank"] is not None and r["correct_rank"] <= 1) / len(cases)
            print(f"  {conf}: {len(cases)} cases, {acc:.1%} accuracy")


def analyse_stability(all_runs):
    """Task 7: Multi-run stability assessment."""
    print(f"\n{'='*80}")
    print("TASK 7: MULTI-RUN STABILITY (n={} runs)".format(len(all_runs)))
    print(f"{'='*80}")

    n_runs = len(all_runs)

    for subset_label, filter_fn in [("ALL", lambda r: True), ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare")]:
        print(f"\n--- {subset_label} ---")

        run_metrics = {"top1": [], "top3": [], "top5": [], "mrr": []}

        for run_idx, run in enumerate(all_runs):
            subset = [r for r in run if filter_fn(r)]
            n = len(subset)
            ranks = [r["correct_rank"] for r in subset]

            t1 = sum(1 for rk in ranks if rk is not None and rk <= 1) / n
            t3 = sum(1 for rk in ranks if rk is not None and rk <= 3) / n
            t5 = sum(1 for rk in ranks if rk is not None and rk <= 5) / n
            mrr = sum(1.0/rk for rk in ranks if rk is not None) / n

            run_metrics["top1"].append(t1)
            run_metrics["top3"].append(t3)
            run_metrics["top5"].append(t5)
            run_metrics["mrr"].append(mrr)

        print(f"{'Metric':<10} {'Run 1':<10} {'Run 2':<10} {'Run 3':<10} {'Mean±SD':<16}")
        print("-" * 56)
        for metric in ["top1", "top3", "top5", "mrr"]:
            vals = run_metrics[metric]
            mean = np.mean(vals)
            sd = np.std(vals)
            val_strs = [f"{v:.1%}" if metric != "mrr" else f"{v:.3f}" for v in vals]
            if metric != "mrr":
                mean_str = f"{mean:.1%}±{sd:.1%}"
            else:
                mean_str = f"{mean:.3f}±{sd:.3f}"
            print(f"{metric:<10} {val_strs[0]:<10} {val_strs[1]:<10} {val_strs[2]:<10} {mean_str:<16}")

    # Per-case agreement across runs
    print(f"\n--- Per-case stability (ultra-rare) ---")
    ur_ids = [r["case_id"] for r in all_runs[0] if r["rarity"] == "ultra-rare"]
    agreement = 0
    total = len(ur_ids)
    for cid in ur_ids:
        top1s = []
        for run in all_runs:
            case = next(r for r in run if r["case_id"] == cid)
            top1_correct = case["correct_rank"] is not None and case["correct_rank"] <= 1
            top1s.append(top1_correct)
        if all(t == top1s[0] for t in top1s):
            agreement += 1

    print(f"  Cases with identical top-1 outcome across all runs: {agreement}/{total} ({agreement/total:.1%})")
    print(f"  Cases with variable outcome: {total - agreement}/{total} ({(total-agreement)/total:.1%})")


def main():
    print("=" * 80)
    print("Confidence Calibration + Multi-Run Stability")
    print("=" * 80)

    # Load components
    stores = {}
    bm25_stores = {}
    index_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "indices"
    for source in ["orphanet", "pubmed"]:
        source_dir = index_dir / source
        if source_dir.exists():
            store = VectorStore.load(source, source_dir)
            stores[source] = store
            if (source_dir / "bm25_config.json").exists():
                bm25_stores[source] = BM25Store.load(source, store.chunks, source_dir)

    embedder = get_embedder("all-MiniLM-L6-v2")

    hpo = HPOOntology()
    hpo_index = Path(__file__).resolve().parents[1] / "data" / "hpo" / "hpo_index.json"
    if hpo_index.exists():
        hpo.load_index(hpo_index)

    all_diseases = load_diseases()
    phenotype_scorer = PhenotypeScorer(hpo, all_diseases)
    disease_chunks_map = {}
    if "orphanet" in stores:
        for chunk in stores["orphanet"].chunks:
            disease_chunks_map.setdefault(chunk.disease_name, []).append(chunk)

    # Optimised retriever: no reranker, with HPO, with BM25
    retriever = MultiSourceRetriever(
        stores=stores, embedder=embedder, bm25_stores=bm25_stores,
        phenotype_scorer=phenotype_scorer, phenotype_weight=1.0,
        disease_chunks_map=disease_chunks_map,
    )

    import anthropic
    client = anthropic.Anthropic()

    query_cache = load_query_cache()

    # Load benchmark
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = BENCHMARK_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    print(f"Loaded {len(all_cases)} benchmark cases")

    # Run 3 passes
    N_RUNS = 3
    all_runs = []
    for run_id in range(1, N_RUNS + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_id}/{N_RUNS}")
        print(f"{'='*80}")
        results = run_single_pass(all_cases, retriever, hpo, client, query_cache, run_id)
        all_runs.append(results)

        # Save incrementally
        out_path = BENCHMARK_DIR / "stability_runs.json"
        with open(out_path, "w") as f:
            json.dump({"runs": all_runs, "n_runs": len(all_runs)}, f, indent=2)
        print(f"  Saved run {run_id} to {out_path}")

    # Analyses
    analyse_confidence(all_runs)
    analyse_stability(all_runs)

    # Save final
    out_path = BENCHMARK_DIR / "stability_runs.json"
    with open(out_path, "w") as f:
        json.dump({"runs": all_runs, "n_runs": N_RUNS}, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
