"""Statistical analyses for the rare disease RAG paper.

Covers:
1. Bootstrap confidence intervals + McNemar's tests
2. Error analysis / failure mode categorisation
3. Knowledge base coverage audit
4. Retrieval-generation decomposition (4-quadrant)
5. Case difficulty predictors
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate.metrics import disease_match

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "indices"


def load_data():
    """Load all benchmark results and case data."""
    with open(BENCHMARK_DIR / "all_approaches_results.json") as f:
        approaches = json.load(f)

    with open(BENCHMARK_DIR / "ablation_results.json") as f:
        ablations = json.load(f)

    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = BENCHMARK_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))

    case_lookup = {c.get("case_id", f"case_{i}"): c for i, c in enumerate(all_cases)}
    return approaches, ablations, all_cases, case_lookup


def load_kb_diseases():
    """Load all disease names from the Orphanet FAISS index."""
    from src.index.vector_store import VectorStore
    orphanet_dir = INDEX_DIR / "orphanet"
    if not orphanet_dir.exists():
        return set()
    store = VectorStore.load("orphanet", orphanet_dir)
    return {chunk.disease_name for chunk in store.chunks}


# =====================================================================
# 1. BOOTSTRAP CONFIDENCE INTERVALS + McNEMAR'S TESTS
# =====================================================================
def bootstrap_ci(results, metric_fn, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for a metric."""
    rng = np.random.default_rng(42)
    n = len(results)
    boot_stats = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = [results[i] for i in idx]
        boot_stats.append(metric_fn(sample))
    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    point = metric_fn(results)
    return point, lo, hi


def top1_metric(results):
    return sum(1 for r in results if r is not None and r <= 1) / len(results)

def top3_metric(results):
    return sum(1 for r in results if r is not None and r <= 3) / len(results)

def top5_metric(results):
    return sum(1 for r in results if r is not None and r <= 5) / len(results)

def mrr_metric(results):
    return sum(1.0/r for r in results if r is not None) / len(results)


def mcnemar_test(ranks_a, ranks_b, k=1):
    """McNemar's test comparing two approaches at top-k.

    Returns chi-squared statistic and p-value.
    Tests whether the two approaches have significantly different accuracy.
    """
    from scipy.stats import chi2

    # Contingency: a_correct & b_correct, a_correct & b_wrong, etc.
    n01 = 0  # A wrong, B correct
    n10 = 0  # A correct, B wrong
    for ra, rb in zip(ranks_a, ranks_b):
        a_correct = ra is not None and ra <= k
        b_correct = rb is not None and rb <= k
        if a_correct and not b_correct:
            n10 += 1
        elif not a_correct and b_correct:
            n01 += 1

    # McNemar's with continuity correction
    if n01 + n10 == 0:
        return 0.0, 1.0  # No discordant pairs

    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value


def run_statistical_tests(approaches):
    """Task 1: Bootstrap CIs and McNemar's tests."""
    print("=" * 80)
    print("1. STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)

    results = approaches["results"]
    approach_names = ["rag", "norag", "hyp_validate", "ensemble"]

    for subset_label, filter_fn in [
        ("ALL (n=55)", lambda r: True),
        ("ULTRA-RARE (n=40)", lambda r: r["rarity"] == "ultra-rare"),
    ]:
        subset = [r for r in results if filter_fn(r)]
        print(f"\n--- {subset_label} ---")

        # Bootstrap CIs
        print(f"\n{'Approach':<18} {'Top-1 (95% CI)':<24} {'Top-3 (95% CI)':<24} {'MRR (95% CI)':<24}")
        print("-" * 90)

        for approach in approach_names:
            ranks = [r[f"{approach}_rank"] for r in subset]
            t1, t1_lo, t1_hi = bootstrap_ci(ranks, top1_metric)
            t3, t3_lo, t3_hi = bootstrap_ci(ranks, top3_metric)
            m, m_lo, m_hi = bootstrap_ci(ranks, mrr_metric)
            print(f"{approach:<18} {t1:.1%} ({t1_lo:.1%}-{t1_hi:.1%}){'':<4} "
                  f"{t3:.1%} ({t3_lo:.1%}-{t3_hi:.1%}){'':<4} "
                  f"{m:.3f} ({m_lo:.3f}-{m_hi:.3f})")

        # McNemar's pairwise tests (top-1)
        print(f"\nMcNemar's test (top-1 accuracy):")
        print(f"{'Comparison':<35} {'n01':<6} {'n10':<6} {'chi2':<8} {'p-value':<10} {'Sig?'}")
        print("-" * 75)

        pairs = [
            ("rag", "norag"),
            ("ensemble", "norag"),
            ("hyp_validate", "norag"),
            ("rag", "ensemble"),
        ]
        for a, b in pairs:
            ranks_a = [r[f"{a}_rank"] for r in subset]
            ranks_b = [r[f"{b}_rank"] for r in subset]
            chi2_stat, p_val = mcnemar_test(ranks_a, ranks_b, k=1)

            # Count discordant pairs for reporting
            n01, n10 = 0, 0
            for ra, rb in zip(ranks_a, ranks_b):
                ac = ra is not None and ra <= 1
                bc = rb is not None and rb <= 1
                if ac and not bc: n10 += 1
                if not ac and bc: n01 += 1

            sig = "*" if p_val < 0.05 else ("†" if p_val < 0.10 else "ns")
            print(f"{a} vs {b:<22} {n01:<6} {n10:<6} {chi2_stat:<8.3f} {p_val:<10.4f} {sig}")

    return


# =====================================================================
# 2 & 3. ERROR ANALYSIS + KNOWLEDGE BASE COVERAGE
# =====================================================================
def run_error_analysis(approaches, all_cases, case_lookup, ablations):
    """Tasks 2 & 3: Error analysis, failure modes, KB coverage."""
    print("\n" + "=" * 80)
    print("2 & 3. ERROR ANALYSIS + KNOWLEDGE BASE COVERAGE")
    print("=" * 80)

    # Load KB disease names
    kb_diseases = load_kb_diseases()
    print(f"Knowledge base contains {len(kb_diseases)} unique disease names")

    results = approaches["results"]

    # Get retrieval recall from ablation data (full_system)
    fs_cases = ablations.get("full_system", {}).get("cases", [])
    # Also get from no_reranker which has retrieval_recall_20
    nr_cases = ablations.get("no_reranker", {}).get("cases", [])
    ret_recall_map = {}
    for c in nr_cases:
        if isinstance(c, dict) and "case_id" in c:
            ret_recall_map[c["case_id"]] = c.get("retrieval_recall_20", False)

    # Check KB coverage for each case
    coverage = {}
    for case in all_cases:
        case_id = case.get("case_id")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        # Check if target disease is in KB
        found = any(disease_match(kb_d, target, aliases) for kb_d in kb_diseases)
        coverage[case_id] = found

    # Categorise ultra-rare cases
    ultra_rare = [r for r in results if r["rarity"] == "ultra-rare"]

    categories = {
        "knowledge_gap": [],       # Disease not in KB
        "retrieval_failure": [],   # In KB, not retrieved
        "generation_failure": [],  # Retrieved but LLM missed
        "rag_success": [],         # RAG got it right (top-5)
        "norag_only": [],          # No-RAG got it but RAG didn't
    }

    print(f"\n{'Case ID':<22} {'Disease':<42} {'InKB':<6} {'Ret':<5} {'RAG':<5} {'NoRAG':<6} {'Category'}")
    print("-" * 130)

    for r in ultra_rare:
        cid = r["case_id"]
        target = r["target"]
        in_kb = coverage.get(cid, False)
        retrieved = ret_recall_map.get(cid, False)
        rag_rank = r["rag_rank"]
        norag_rank = r["norag_rank"]

        rag_hit = rag_rank is not None and rag_rank <= 5
        norag_hit = norag_rank is not None and norag_rank <= 5

        if not in_kb:
            cat = "knowledge_gap"
        elif not retrieved and not rag_hit:
            cat = "retrieval_failure"
        elif retrieved and not rag_hit:
            cat = "generation_failure"
        elif rag_hit:
            cat = "rag_success"
        elif norag_hit and not rag_hit:
            cat = "norag_only"
        else:
            cat = "retrieval_failure"  # not retrieved, not hit

        categories[cat].append(r)

        rag_str = str(rag_rank) if rag_rank else "MISS"
        norag_str = str(norag_rank) if norag_rank else "MISS"
        ret_str = "Y" if retrieved else "N"
        kb_str = "Y" if in_kb else "N"
        print(f"{cid:<22} {target[:40]:<42} {kb_str:<6} {ret_str:<5} {rag_str:<5} {norag_str:<6} {cat}")

    print(f"\n--- Failure Mode Summary (Ultra-Rare, n={len(ultra_rare)}) ---")
    for cat, cases in categories.items():
        pct = len(cases) / len(ultra_rare) * 100
        print(f"  {cat:<25} {len(cases):>3} ({pct:.1f}%)")

    # KB coverage summary
    ur_covered = sum(1 for r in ultra_rare if coverage.get(r["case_id"], False))
    all_covered = sum(1 for c in all_cases if coverage.get(c.get("case_id"), False))
    print(f"\n--- Knowledge Base Coverage ---")
    print(f"  All cases:       {all_covered}/{len(all_cases)} ({all_covered/len(all_cases):.1%})")
    print(f"  Ultra-rare:      {ur_covered}/{len(ultra_rare)} ({ur_covered/len(ultra_rare):.1%})")

    # Accuracy on COVERED diseases only
    ur_covered_cases = [r for r in ultra_rare if coverage.get(r["case_id"], False)]
    if ur_covered_cases:
        for approach in ["rag", "norag", "ensemble"]:
            ranks = [r[f"{approach}_rank"] for r in ur_covered_cases]
            t1 = sum(1 for rk in ranks if rk is not None and rk <= 1) / len(ranks)
            t5 = sum(1 for rk in ranks if rk is not None and rk <= 5) / len(ranks)
            print(f"  {approach} on covered ultra-rare (n={len(ur_covered_cases)}): top-1={t1:.1%}, top-5={t5:.1%}")

    return coverage, categories


# =====================================================================
# 5. RETRIEVAL-GENERATION DECOMPOSITION
# =====================================================================
def run_decomposition(approaches, ablations):
    """Task 5: 4-quadrant retrieval vs generation analysis."""
    print("\n" + "=" * 80)
    print("5. RETRIEVAL-GENERATION DECOMPOSITION")
    print("=" * 80)

    results = approaches["results"]

    # Get retrieval recall from no_reranker ablation (has ret data for full system equivalent)
    nr_cases = ablations.get("no_reranker", {}).get("cases", [])
    ret_map = {c["case_id"]: c.get("retrieval_recall_20", False) for c in nr_cases if isinstance(c, dict)}

    # 4 quadrants for RAG approach
    quadrants = {
        "retrieved_correct": [],      # Retrieved + RAG correct (top-5)
        "retrieved_missed": [],       # Retrieved but RAG missed
        "not_retrieved_correct": [],  # Not retrieved but RAG correct (LLM knowledge)
        "not_retrieved_missed": [],   # Neither retrieved nor correct
    }

    for r in results:
        cid = r["case_id"]
        if r["rarity"] != "ultra-rare":
            continue

        retrieved = ret_map.get(cid, False)
        rag_correct = r["rag_rank"] is not None and r["rag_rank"] <= 5

        if retrieved and rag_correct:
            quadrants["retrieved_correct"].append(r)
        elif retrieved and not rag_correct:
            quadrants["retrieved_missed"].append(r)
        elif not retrieved and rag_correct:
            quadrants["not_retrieved_correct"].append(r)
        else:
            quadrants["not_retrieved_missed"].append(r)

    n_ur = sum(len(v) for v in quadrants.values())
    print(f"\nUltra-Rare Cases (n={n_ur}) — RAG approach, top-5:")
    print(f"\n{'':>30} | {'Generated Correct':>18} | {'Generated Wrong':>16} |")
    print(f"{'-'*30}-+-{'-'*18}-+-{'-'*16}-+")
    rc = len(quadrants["retrieved_correct"])
    rm = len(quadrants["retrieved_missed"])
    nrc = len(quadrants["not_retrieved_correct"])
    nrm = len(quadrants["not_retrieved_missed"])
    print(f"{'Retrieved (Ret@20)':>30} | {rc:>8} ({rc/n_ur:.1%})    | {rm:>6} ({rm/n_ur:.1%})    |")
    print(f"{'Not Retrieved':>30} | {nrc:>8} ({nrc/n_ur:.1%})    | {nrm:>6} ({nrm/n_ur:.1%})    |")

    print(f"\nInterpretation:")
    print(f"  RAG value-add (retrieved + correct):     {rc} cases — retrieval directly helped diagnosis")
    print(f"  Retrieval gap (retrieved + wrong):        {rm} cases — evidence found but LLM couldn't use it")
    print(f"  LLM parametric knowledge (not ret + correct): {nrc} cases — LLM knew answer without retrieval")
    print(f"  Total failure (neither):                 {nrm} cases — neither retrieval nor LLM knowledge sufficient")

    # Also do for No-RAG comparison
    print(f"\n--- Cross-approach comparison (top-5) ---")
    rag_only = norag_only = both = neither = 0
    for r in results:
        if r["rarity"] != "ultra-rare":
            continue
        rag_hit = r["rag_rank"] is not None and r["rag_rank"] <= 5
        norag_hit = r["norag_rank"] is not None and r["norag_rank"] <= 5
        if rag_hit and norag_hit: both += 1
        elif rag_hit and not norag_hit: rag_only += 1
        elif not rag_hit and norag_hit: norag_only += 1
        else: neither += 1

    print(f"  Both RAG and No-RAG correct:  {both}")
    print(f"  RAG only correct:             {rag_only} (unique RAG contribution)")
    print(f"  No-RAG only correct:          {norag_only} (RAG hurt these)")
    print(f"  Neither correct:              {neither}")

    return quadrants


# =====================================================================
# 9. CASE DIFFICULTY PREDICTORS
# =====================================================================
def run_difficulty_analysis(approaches, all_cases, case_lookup):
    """Task 9: What predicts diagnostic difficulty?"""
    print("\n" + "=" * 80)
    print("9. CASE DIFFICULTY PREDICTORS")
    print("=" * 80)

    results = approaches["results"]
    ultra_rare = [r for r in results if r["rarity"] == "ultra-rare"]

    # Features per case
    features = []
    for r in ultra_rare:
        case = case_lookup.get(r["case_id"], {})
        vignette = case.get("clinical_vignette", "")
        hpo_count = len(case.get("key_discriminating_features", []))
        vignette_len = len(vignette.split())
        n_aliases = len(case.get("aliases", []))

        # Any approach got it?
        any_correct = any(
            r[a] is not None and r[a] <= 5
            for a in ["rag_rank", "norag_rank", "hyp_validate_rank", "ensemble_rank"]
        )

        features.append({
            "case_id": r["case_id"],
            "target": r["target"],
            "vignette_words": vignette_len,
            "n_discriminating_features": hpo_count,
            "n_aliases": n_aliases,
            "any_approach_top5": any_correct,
            "rag_rank": r["rag_rank"],
            "norag_rank": r["norag_rank"],
        })

    # Compare solved vs unsolved
    solved = [f for f in features if f["any_approach_top5"]]
    unsolved = [f for f in features if not f["any_approach_top5"]]

    print(f"\nSolved by any approach (top-5): {len(solved)}/{len(features)}")
    print(f"Unsolved by all approaches:     {len(unsolved)}/{len(features)}")

    for metric in ["vignette_words", "n_discriminating_features", "n_aliases"]:
        s_vals = [f[metric] for f in solved]
        u_vals = [f[metric] for f in unsolved]
        s_mean = np.mean(s_vals) if s_vals else 0
        u_mean = np.mean(u_vals) if u_vals else 0
        print(f"\n  {metric}:")
        print(f"    Solved:   mean={s_mean:.1f}, median={np.median(s_vals) if s_vals else 0:.1f}")
        print(f"    Unsolved: mean={u_mean:.1f}, median={np.median(u_vals) if u_vals else 0:.1f}")

    # List unsolved cases
    print(f"\n--- Unsolved Ultra-Rare Cases ---")
    for f in unsolved:
        print(f"  {f['case_id']}: {f['target']}")

    return features


def main():
    approaches, ablations, all_cases, case_lookup = load_data()

    # 1. Statistical tests
    run_statistical_tests(approaches)

    # 2 & 3. Error analysis + KB coverage
    coverage, categories = run_error_analysis(approaches, all_cases, case_lookup, ablations)

    # 5. Retrieval-generation decomposition
    quadrants = run_decomposition(approaches, ablations)

    # 9. Case difficulty
    features = run_difficulty_analysis(approaches, all_cases, case_lookup)

    # Save all analyses
    output = {
        "failure_categories": {k: [r["case_id"] for r in v] for k, v in categories.items()},
        "kb_coverage": coverage,
        "quadrants": {k: [r["case_id"] for r in v] for k, v in quadrants.items()},
    }
    out_path = BENCHMARK_DIR / "statistical_analyses.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved analyses to {out_path}")


if __name__ == "__main__":
    main()
