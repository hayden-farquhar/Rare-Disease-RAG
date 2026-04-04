"""Run RAG+HyDE and No-RAG on the 30 new cases, then compute combined metrics."""

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.retrieve.retriever import MultiSourceRetriever
from src.retrieve.bm25_store import BM25Store
from src.retrieve.query_processor import process_query
from src.retrieve.context_assembler import assemble_context
from src.retrieve.phenotype_scorer import PhenotypeScorer
from src.generate.generator import generate_diagnosis, generate_diagnosis_no_rag
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match

import numpy as np
from scipy.stats import chi2 as chi2_dist

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
HYDE_CACHE_PATH = BENCHMARK_DIR / "hyde_cache.json"


def load_hyde_cache():
    if HYDE_CACHE_PATH.exists():
        with open(HYDE_CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_hyde_cache(cache):
    with open(HYDE_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_hyde_processed_query(vignette, hpo, client, cache):
    from src.retrieve.query_processor import ProcessedQuery
    key = vignette[:200]
    if key in cache:
        return ProcessedQuery(**cache[key])
    processed = process_query(vignette, hpo_ontology=hpo, use_llm=True,
                              client=client, use_hyde=True)
    if processed.extracted_phenotypes:
        processed.retrieval_queries.append(
            "rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5])
        )
    cache[key] = processed.model_dump()
    save_hyde_cache(cache)
    return processed


def retrieval_recall(retrieved, target, aliases, k=20):
    for chunk in retrieved[:k]:
        if disease_match(chunk.disease_name, target, aliases):
            return True
    return False


def bootstrap_ci(ranks, metric_fn, n_boot=10000):
    rng = np.random.default_rng(42)
    n = len(ranks)
    stats = [metric_fn([ranks[i] for i in rng.integers(0, n, size=n)]) for _ in range(n_boot)]
    return np.percentile(stats, 2.5), np.percentile(stats, 97.5)


def top1_fn(ranks):
    return sum(1 for r in ranks if r is not None and r <= 1) / len(ranks)


def mcnemar(ranks_a, ranks_b, k=1):
    n01 = n10 = 0
    for ra, rb in zip(ranks_a, ranks_b):
        ac = ra is not None and ra <= k
        bc = rb is not None and rb <= k
        if ac and not bc: n10 += 1
        if not ac and bc: n01 += 1
    if n01 + n10 == 0:
        return 0.0, 1.0
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - chi2_dist.cdf(chi2_stat, df=1)
    return chi2_stat, p


def main():
    print("=" * 70)
    print("Expanded Benchmark — 30 New Cases (RAG+HyDE + No-RAG)")
    print("=" * 70)

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

    retriever = MultiSourceRetriever(
        stores=stores, embedder=embedder, bm25_stores=bm25_stores,
        phenotype_scorer=phenotype_scorer, phenotype_weight=1.0,
        disease_chunks_map=disease_chunks_map,
    )

    import anthropic
    client = anthropic.Anthropic()
    hyde_cache = load_hyde_cache()

    # Load NEW cases only
    with open(BENCHMARK_DIR / "ultra_rare_cases_new.json") as f:
        new_cases = json.load(f)
    print(f"Loaded {len(new_cases)} new cases")

    results = []
    for i, case in enumerate(new_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        rarity = case.get("rarity", "ultra-rare")
        vignette = case["clinical_vignette"]

        # === RAG + HyDE (no reranker) ===
        processed = get_hyde_processed_query(vignette, hpo, client, hyde_cache)
        queries = processed.retrieval_queries or [vignette[:500]]
        query_hpo_ids = []
        if processed.hpo_candidates:
            query_hpo_ids = [h["hpo_id"] for h in processed.hpo_candidates if h.get("score", 0) > 0.5]

        retrieved = retriever.retrieve(
            queries=queries, top_k=50, per_query_k=25,
            query_hpo_ids=query_hpo_ids,
            hyde_document=processed.hyde_document,
        )
        context = assemble_context(retrieved, max_tokens=6000)
        rag_output = generate_diagnosis(vignette=vignette, context=context, client=client)
        rag_preds = [dx.disease_name for dx in rag_output.differential_diagnosis]

        rag_rank = None
        for j, p in enumerate(rag_preds):
            if disease_match(p, target, aliases):
                rag_rank = j + 1
                break

        ret_recall = retrieval_recall(retrieved, target, aliases, k=20)

        # === No-RAG ===
        norag_output = generate_diagnosis_no_rag(vignette=vignette, client=client)
        norag_preds = [dx.disease_name for dx in norag_output.differential_diagnosis]

        norag_rank = None
        for j, p in enumerate(norag_preds):
            if disease_match(p, target, aliases):
                norag_rank = j + 1
                break

        results.append({
            "case_id": case_id,
            "target": target,
            "rarity": rarity,
            "rag_hyde_rank": rag_rank,
            "norag_rank": norag_rank,
            "retrieval_recall_20": ret_recall,
        })

        rag_str = str(rag_rank) if rag_rank else "MISS"
        norag_str = str(norag_rank) if norag_rank else "MISS"
        ret_str = "Y" if ret_recall else "N"
        print(f"  [{i+1}/{len(new_cases)}] {case_id}: RAG={rag_str} NoRAG={norag_str} ret={ret_str}")

        time.sleep(0.5)

    # Save new case results
    out_path = BENCHMARK_DIR / "expanded_benchmark_new_cases.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # === COMBINED METRICS ===
    # Load original results
    with open(BENCHMARK_DIR / "hyde_benchmark_results.json") as f:
        hyde_orig = json.load(f)["results"]
    with open(BENCHMARK_DIR / "all_approaches_results.json") as f:
        orig_approaches = json.load(f)["results"]

    # Build combined dataset
    # Original 55: use HyDE results for RAG, original for No-RAG
    combined = []
    for ho, oa in zip(hyde_orig, orig_approaches):
        combined.append({
            "case_id": ho["case_id"],
            "target": ho["target"],
            "rarity": ho["rarity"],
            "rag_hyde_rank": ho["rank"],
            "norag_rank": oa["norag_rank"],
            "retrieval_recall_20": ho.get("retrieval_recall_20", False),
        })
    # Add new 30
    combined.extend(results)

    print(f"\n{'='*70}")
    print(f"COMBINED RESULTS — {len(combined)} cases (55 original + {len(results)} new)")
    print(f"{'='*70}")

    for subset_label, filter_fn in [
        ("ALL", lambda r: True),
        ("WELL-KNOWN", lambda r: r["rarity"] != "ultra-rare"),
        ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare"),
    ]:
        subset = [r for r in combined if filter_fn(r)]
        if not subset:
            continue
        n = len(subset)

        rag_ranks = [r["rag_hyde_rank"] for r in subset]
        norag_ranks = [r["norag_rank"] for r in subset]

        rag_t1 = sum(1 for r in rag_ranks if r is not None and r <= 1) / n
        rag_t3 = sum(1 for r in rag_ranks if r is not None and r <= 3) / n
        rag_t5 = sum(1 for r in rag_ranks if r is not None and r <= 5) / n
        rag_mrr = sum(1.0/r for r in rag_ranks if r is not None) / n

        norag_t1 = sum(1 for r in norag_ranks if r is not None and r <= 1) / n
        norag_t3 = sum(1 for r in norag_ranks if r is not None and r <= 3) / n
        norag_t5 = sum(1 for r in norag_ranks if r is not None and r <= 5) / n
        norag_mrr = sum(1.0/r for r in norag_ranks if r is not None) / n

        ret = sum(1 for r in subset if r.get("retrieval_recall_20")) / n

        # Bootstrap CIs for ultra-rare
        rag_lo, rag_hi = bootstrap_ci(rag_ranks, top1_fn)
        norag_lo, norag_hi = bootstrap_ci(norag_ranks, top1_fn)

        # McNemar's
        chi2_stat, p_val = mcnemar(rag_ranks, norag_ranks, k=1)

        print(f"\n--- {subset_label} (n={n}) ---")
        print(f"{'Approach':<20} {'Top-1 (95% CI)':<26} {'Top-3':<10} {'Top-5':<10} {'MRR':<10}")
        print("-" * 76)
        print(f"{'RAG+HyDE':<20} {rag_t1:.1%} ({rag_lo:.1%}-{rag_hi:.1%}){'':<6} {rag_t3:<10.1%} {rag_t5:<10.1%} {rag_mrr:<10.3f}")
        print(f"{'No-RAG':<20} {norag_t1:.1%} ({norag_lo:.1%}-{norag_hi:.1%}){'':<6} {norag_t3:<10.1%} {norag_t5:<10.1%} {norag_mrr:<10.3f}")
        print(f"  Ret@20: {ret:.1%}")
        print(f"  McNemar's (top-1): chi2={chi2_stat:.3f}, p={p_val:.4f}")
        delta = rag_t1 - norag_t1
        print(f"  RAG advantage: +{delta:.1%}")

    # Save combined
    out_path = BENCHMARK_DIR / "expanded_benchmark_combined.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
