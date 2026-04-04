"""Run ablation studies — disable each component one at a time.

Component ablations: remove reranker, LLM queries, HPO scorer, BM25
Source ablations: Orphanet-only, PubMed-only
Baseline: full_system results loaded from completed 4-approach comparison.

Caches LLM query expansion results to avoid redundant API calls.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.index.chunker import Chunk
from src.retrieve.retriever import MultiSourceRetriever, CrossEncoderReranker, RetrievedChunk
from src.retrieve.bm25_store import BM25Store
from src.retrieve.query_processor import process_query
from src.retrieve.context_assembler import assemble_context
from src.retrieve.phenotype_scorer import PhenotypeScorer
from src.generate.generator import generate_diagnosis
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match, top_k_accuracy, mean_reciprocal_rank


# ---------------------------------------------------------------------------
# Query expansion cache — avoids re-calling the LLM for identical vignettes
# ---------------------------------------------------------------------------
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "query_cache.json"


def load_query_cache() -> dict:
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_query_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_processed_query(vignette: str, hpo, client, cache: dict):
    """Return cached ProcessedQuery data or compute and cache it."""
    key = vignette[:200]  # first 200 chars as key (unique enough)
    if key in cache:
        from src.retrieve.query_processor import ProcessedQuery
        return ProcessedQuery(**cache[key])

    processed = process_query(vignette, hpo_ontology=hpo, use_llm=True, client=client)

    # Add disease-name augmentation query
    if processed.extracted_phenotypes:
        processed.retrieval_queries.append(
            "rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5])
        )

    cache[key] = processed.model_dump()
    return processed


# ---------------------------------------------------------------------------
# RAG pipeline with configurable components
# ---------------------------------------------------------------------------
def run_rag_pipeline(case, retriever, reranker, hpo, client, query_cache,
                     use_llm_queries=True, use_reranker=True, use_hpo=True):
    """Run RAG pipeline with configurable components. Returns (output, retrieved_chunks)."""
    vignette = case["clinical_vignette"]

    # --- Query processing ---
    if use_llm_queries:
        processed = get_processed_query(vignette, hpo, client, query_cache)
        queries = processed.retrieval_queries or [vignette[:500]]
    else:
        from src.retrieve.query_processor import ProcessedQuery
        queries = [vignette[:500]]
        key_features = case.get("key_discriminating_features", [])
        if key_features:
            queries.append(" ".join(key_features))
        processed = ProcessedQuery(original_vignette=vignette)

    # --- HPO IDs for phenotype channel ---
    query_hpo_ids = []
    if use_hpo and processed.hpo_candidates:
        query_hpo_ids = [
            h["hpo_id"] for h in processed.hpo_candidates if h.get("score", 0) > 0.5
        ]

    # --- Retrieval ---
    retrieved = retriever.retrieve(
        queries=queries, top_k=50, per_query_k=25,
        query_hpo_ids=query_hpo_ids if use_hpo else None,
    )

    # --- Reranking ---
    if use_reranker and reranker and retrieved:
        retrieved = reranker.rerank(query=vignette, chunks=retrieved, top_k=20)

    # --- Context + Generation ---
    context = assemble_context(retrieved, max_tokens=6000)
    output = generate_diagnosis(vignette=vignette, context=context, client=client)

    return output, retrieved


# ---------------------------------------------------------------------------
# Retrieval recall: did the correct disease appear in retrieved chunks?
# ---------------------------------------------------------------------------
def retrieval_recall(retrieved: list[RetrievedChunk], target: str,
                     aliases: list[str], k: int = 20) -> bool:
    """Check if any retrieved chunk (top-k) mentions the target disease."""
    for chunk in retrieved[:k]:
        if disease_match(chunk.disease_name, target, aliases):
            return True
    return False


def main():
    print("=" * 70)
    print("Ablation Studies — Component & Source Contribution")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load shared components
    # ------------------------------------------------------------------
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

    reranker = CrossEncoderReranker()

    import anthropic
    client = anthropic.Anthropic()

    # Load query expansion cache
    query_cache = load_query_cache()

    # ------------------------------------------------------------------
    # Load benchmark cases
    # ------------------------------------------------------------------
    benchmark_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = benchmark_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    print(f"Loaded {len(all_cases)} benchmark cases")

    # ------------------------------------------------------------------
    # Load full_system baseline from completed 4-approach comparison
    # ------------------------------------------------------------------
    prev_path = benchmark_dir / "all_approaches_results.json"
    full_system_results = None
    if prev_path.exists():
        with open(prev_path) as f:
            prev = json.load(f)
        full_system_results = {r["case_id"]: r["rag_rank"] for r in prev["results"]}
        print(f"Loaded full_system baseline from previous run ({len(full_system_results)} cases)")

    # ------------------------------------------------------------------
    # Ablation configurations
    # ------------------------------------------------------------------
    # Each config: what changes from full system
    ablation_configs = {
        # Component ablations — remove ONE component at a time
        "no_reranker": {
            "use_llm_queries": True, "use_reranker": False, "use_hpo": True,
            "use_bm25": True, "sources": ["orphanet", "pubmed"],
        },
        "no_llm_queries": {
            "use_llm_queries": False, "use_reranker": True, "use_hpo": True,
            "use_bm25": True, "sources": ["orphanet", "pubmed"],
        },
        "no_hpo_scorer": {
            "use_llm_queries": True, "use_reranker": True, "use_hpo": False,
            "use_bm25": True, "sources": ["orphanet", "pubmed"],
        },
        "no_bm25": {
            "use_llm_queries": True, "use_reranker": True, "use_hpo": True,
            "use_bm25": False, "sources": ["orphanet", "pubmed"],
        },
        # Source ablations — single knowledge base
        "orphanet_only": {
            "use_llm_queries": True, "use_reranker": True, "use_hpo": True,
            "use_bm25": True, "sources": ["orphanet"],
        },
        "pubmed_only": {
            "use_llm_queries": True, "use_reranker": True, "use_hpo": False,
            "use_bm25": True, "sources": ["pubmed"],
        },
    }

    all_results = {}

    # ------------------------------------------------------------------
    # Run each ablation
    # ------------------------------------------------------------------
    for config_name, config in ablation_configs.items():
        print(f"\n{'='*70}")
        print(f"ABLATION: {config_name}")
        active_sources = config["sources"]
        print(f"  Sources: {active_sources} | LLM queries: {config['use_llm_queries']} | "
              f"Reranker: {config['use_reranker']} | HPO: {config['use_hpo']} | "
              f"BM25: {config['use_bm25']}")
        print(f"{'='*70}")

        # Build retriever for this config
        config_stores = {s: stores[s] for s in active_sources if s in stores}
        config_bm25 = {s: bm25_stores[s] for s in active_sources if s in bm25_stores} if config["use_bm25"] else {}
        config_scorer = phenotype_scorer if config["use_hpo"] else None
        config_disease_map = disease_chunks_map if config["use_hpo"] else {}

        retriever = MultiSourceRetriever(
            stores=config_stores, embedder=embedder, bm25_stores=config_bm25,
            phenotype_scorer=config_scorer,
            phenotype_weight=1.0 if config["use_hpo"] else 0,
            disease_chunks_map=config_disease_map,
        )

        predictions = []
        ground_truths = []
        aliases_list = []
        case_results = []
        retrieval_recalls = []

        for i, case in enumerate(all_cases):
            case_id = case.get("case_id", f"case_{i}")
            target = case["final_diagnosis"]
            aliases = case.get("aliases", [])
            rarity = case.get("rarity", "unknown")

            output, retrieved = run_rag_pipeline(
                case, retriever, reranker, hpo, client, query_cache,
                use_llm_queries=config["use_llm_queries"],
                use_reranker=config["use_reranker"],
                use_hpo=config["use_hpo"],
            )

            preds = [dx.disease_name for dx in output.differential_diagnosis]
            rank = None
            for j, p in enumerate(preds):
                if disease_match(p, target, aliases):
                    rank = j + 1
                    break

            ret_recall = retrieval_recall(retrieved, target, aliases, k=20)

            predictions.append(preds)
            ground_truths.append(target)
            aliases_list.append(aliases)
            retrieval_recalls.append(ret_recall)
            case_results.append({
                "case_id": case_id, "rank": rank, "rarity": rarity,
                "retrieval_recall_20": ret_recall,
            })

            rank_str = str(rank) if rank else "MISS"
            ret_str = "Y" if ret_recall else "N"
            print(f"  [{i+1}/{len(all_cases)}] {case_id}: rank={rank_str} ret={ret_str}")

            time.sleep(0.5)

        # Save query cache after each config (incremental)
        save_query_cache(query_cache)

        # --- Compute metrics ---
        metrics = {}
        for subset_name, filter_fn in [
            ("all", lambda r: True),
            ("well_known", lambda r: r["rarity"] != "ultra-rare"),
            ("ultra_rare", lambda r: r["rarity"] == "ultra-rare"),
        ]:
            indices = [i for i, r in enumerate(case_results) if filter_fn(r)]
            if not indices:
                continue
            sub_preds = [predictions[i] for i in indices]
            sub_gt = [ground_truths[i] for i in indices]
            sub_al = [aliases_list[i] for i in indices]
            sub_ret = [retrieval_recalls[i] for i in indices]

            metrics[subset_name] = {
                f"top{k}": top_k_accuracy(sub_preds, sub_gt, sub_al, k=k)
                for k in [1, 3, 5]
            }
            metrics[subset_name]["mrr"] = mean_reciprocal_rank(sub_preds, sub_gt, sub_al)
            metrics[subset_name]["retrieval_recall_20"] = sum(sub_ret) / len(sub_ret)
            metrics[subset_name]["n"] = len(indices)

        all_results[config_name] = {"metrics": metrics, "cases": case_results}

        ur = metrics.get("ultra_rare", {})
        ov = metrics.get("all", {})
        print(f"\n  Ultra-rare: top-1={ur.get('top1', 0):.1%}, ret_recall={ur.get('retrieval_recall_20', 0):.1%}")
        print(f"  Overall:    top-1={ov.get('top1', 0):.1%}, ret_recall={ov.get('retrieval_recall_20', 0):.1%}")

    # ------------------------------------------------------------------
    # Inject full_system baseline from 4-approach comparison
    # ------------------------------------------------------------------
    if full_system_results:
        fs_cases = []
        fs_preds_dummy = []  # We don't have full predictions, only ranks
        for case in all_cases:
            cid = case.get("case_id")
            rank = full_system_results.get(cid)
            fs_cases.append({
                "case_id": cid, "rank": rank,
                "rarity": case.get("rarity", "unknown"),
            })

        # Compute metrics from ranks directly
        fs_metrics = {}
        for subset_name, filter_fn in [
            ("all", lambda r: True),
            ("well_known", lambda r: r["rarity"] != "ultra-rare"),
            ("ultra_rare", lambda r: r["rarity"] == "ultra-rare"),
        ]:
            subset = [r for r in fs_cases if filter_fn(r)]
            if not subset:
                continue
            n = len(subset)
            fs_metrics[subset_name] = {
                f"top{k}": sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k) / n
                for k in [1, 3, 5]
            }
            ranks = [r["rank"] for r in subset]
            fs_metrics[subset_name]["mrr"] = sum(
                1.0 / r for r in ranks if r is not None
            ) / n
            fs_metrics[subset_name]["n"] = n

        all_results["full_system"] = {"metrics": fs_metrics, "cases": fs_cases}

    # ==================================================================
    # SUMMARY TABLE
    # ==================================================================
    print(f"\n{'='*80}")
    print("ABLATION STUDY — SUMMARY")
    print(f"{'='*80}")

    # Order: full_system first, then ablations
    display_order = ["full_system"] + [c for c in ablation_configs]

    for subset_label, subset_key in [("ALL CASES", "all"), ("WELL-KNOWN", "well_known"), ("ULTRA-RARE", "ultra_rare")]:
        print(f"\n--- {subset_label} ---")
        header = f"{'Config':<20} {'n':<5} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'MRR':<8} {'Ret@20':<8} {'Delta':<8}"
        print(header)
        print("-" * len(header))

        baseline_top1 = None
        for config_name in display_order:
            if config_name not in all_results:
                continue
            m = all_results[config_name]["metrics"].get(subset_key, {})
            if not m:
                continue
            n = m.get("n", "?")
            t1 = m.get("top1", 0)
            t3 = m.get("top3", 0)
            t5 = m.get("top5", 0)
            mrr = m.get("mrr", 0)
            ret = m.get("retrieval_recall_20", None)
            ret_str = f"{ret:.1%}" if ret is not None else "—"

            if baseline_top1 is None:
                baseline_top1 = t1
                delta_str = "—"
            else:
                delta = t1 - baseline_top1
                delta_str = f"{delta:+.1%}"

            print(f"{config_name:<20} {n:<5} {t1:<8.1%} {t3:<8.1%} {t5:<8.1%} {mrr:<8.3f} {ret_str:<8} {delta_str:<8}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = benchmark_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
