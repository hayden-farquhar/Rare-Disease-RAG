"""Run remaining source ablations: finish orphanet_only (from case 35) + full pubmed_only."""

import json
import sys
import time
from pathlib import Path

# Unbuffered output
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

# Reuse query cache
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "query_cache.json"

def load_query_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}

def get_processed_query(vignette, hpo, client, cache):
    from src.retrieve.query_processor import process_query, ProcessedQuery
    key = vignette[:200]
    if key in cache:
        return ProcessedQuery(**cache[key])
    processed = process_query(vignette, hpo_ontology=hpo, use_llm=True, client=client)
    if processed.extracted_phenotypes:
        processed.retrieval_queries.append(
            "rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5])
        )
    cache[key] = processed.model_dump()
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
    return processed

def retrieval_recall(retrieved, target, aliases, k=20):
    for chunk in retrieved[:k]:
        if disease_match(chunk.disease_name, target, aliases):
            return True
    return False

def run_config(config_name, config, all_cases, stores, bm25_stores, embedder,
               reranker, hpo, phenotype_scorer, disease_chunks_map, client,
               query_cache, skip_n=0):
    """Run one ablation config. skip_n skips first N cases (for resuming)."""
    active_sources = config["sources"]
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

    case_results = []
    for i, case in enumerate(all_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        rarity = case.get("rarity", "unknown")
        vignette = case["clinical_vignette"]

        if i < skip_n:
            case_results.append(None)  # placeholder
            continue

        # Query processing
        if config["use_llm_queries"]:
            processed = get_processed_query(vignette, hpo, client, query_cache)
            queries = processed.retrieval_queries or [vignette[:500]]
        else:
            from src.retrieve.query_processor import ProcessedQuery
            queries = [vignette[:500]]
            processed = ProcessedQuery(original_vignette=vignette)

        query_hpo_ids = []
        if config["use_hpo"] and processed.hpo_candidates:
            query_hpo_ids = [h["hpo_id"] for h in processed.hpo_candidates if h.get("score", 0) > 0.5]

        retrieved = retriever.retrieve(
            queries=queries, top_k=50, per_query_k=25,
            query_hpo_ids=query_hpo_ids if config["use_hpo"] else None,
        )

        if config["use_reranker"] and reranker and retrieved:
            retrieved = reranker.rerank(query=vignette, chunks=retrieved, top_k=20)

        context = assemble_context(retrieved, max_tokens=6000)
        output = generate_diagnosis(vignette=vignette, context=context, client=client)

        preds = [dx.disease_name for dx in output.differential_diagnosis]
        rank = None
        for j, p in enumerate(preds):
            if disease_match(p, target, aliases):
                rank = j + 1
                break

        ret_recall = retrieval_recall(retrieved, target, aliases, k=20)
        case_results.append({
            "case_id": case_id, "rank": rank, "rarity": rarity,
            "retrieval_recall_20": ret_recall,
        })

        rank_str = str(rank) if rank else "MISS"
        ret_str = "Y" if ret_recall else "N"
        print(f"  [{i+1}/{len(all_cases)}] {case_id}: rank={rank_str} ret={ret_str}")
        time.sleep(0.5)

    return case_results


def main():
    print("=" * 70)
    print("Remaining Source Ablations")
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

    reranker = CrossEncoderReranker()

    import anthropic
    client = anthropic.Anthropic()

    query_cache = load_query_cache()

    benchmark_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = benchmark_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    print(f"Loaded {len(all_cases)} benchmark cases")

    # Load existing ablation results
    abl_path = benchmark_dir / "ablation_results.json"
    with open(abl_path) as f:
        all_results = json.load(f)

    # --- orphanet_only: resume from case 35 (0-indexed: 34) ---
    print(f"\n{'='*70}")
    print("ABLATION: orphanet_only (resuming from case 35)")
    print(f"{'='*70}")

    orphanet_config = {
        "use_llm_queries": True, "use_reranker": True, "use_hpo": True,
        "use_bm25": True, "sources": ["orphanet"],
    }
    orphanet_results = run_config(
        "orphanet_only", orphanet_config, all_cases, stores, bm25_stores,
        embedder, reranker, hpo, phenotype_scorer, disease_chunks_map,
        client, query_cache, skip_n=34,
    )

    # Merge with existing partial results
    existing_cases = all_results["orphanet_only"]["cases"]
    for i, result in enumerate(orphanet_results):
        if result is not None:
            if i < len(existing_cases):
                existing_cases[i] = result
            else:
                existing_cases.append(result)
    all_results["orphanet_only"]["cases"] = existing_cases
    all_results["orphanet_only"]["complete"] = len(existing_cases) == 55

    # --- pubmed_only: full run ---
    print(f"\n{'='*70}")
    print("ABLATION: pubmed_only (full run)")
    print(f"{'='*70}")

    pubmed_config = {
        "use_llm_queries": True, "use_reranker": True, "use_hpo": False,
        "use_bm25": True, "sources": ["pubmed"],
    }
    pubmed_results = run_config(
        "pubmed_only", pubmed_config, all_cases, stores, bm25_stores,
        embedder, reranker, hpo, phenotype_scorer, disease_chunks_map,
        client, query_cache, skip_n=0,
    )
    all_results["pubmed_only"] = {
        "cases": pubmed_results,
        "complete": True,
    }

    # Recompute metrics for all configs
    def compute_metrics(case_results):
        valid = [r for r in case_results if r is not None]
        metrics = {}
        for subset_name, filter_fn in [
            ("all", lambda r: True),
            ("well_known", lambda r: r["rarity"] != "ultra-rare"),
            ("ultra_rare", lambda r: r["rarity"] == "ultra-rare"),
        ]:
            subset = [r for r in valid if filter_fn(r)]
            if not subset:
                continue
            n = len(subset)
            metrics[subset_name] = {
                f"top{k}": sum(1 for r in subset if r["rank"] is not None and r["rank"] <= k) / n
                for k in [1, 3, 5]
            }
            ranks = [r["rank"] for r in subset]
            metrics[subset_name]["mrr"] = sum(1.0/r for r in ranks if r is not None) / n
            has_ret = any("retrieval_recall_20" in r for r in subset)
            if has_ret:
                metrics[subset_name]["retrieval_recall_20"] = sum(
                    1 for r in subset if r.get("retrieval_recall_20", False)
                ) / n
            metrics[subset_name]["n"] = n
        return metrics

    for config_name in ["orphanet_only", "pubmed_only"]:
        all_results[config_name]["metrics"] = compute_metrics(all_results[config_name]["cases"])

    # Save
    with open(abl_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved updated results to {abl_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SOURCE ABLATION RESULTS")
    print(f"{'='*70}")

    display = ["full_system", "orphanet_only", "pubmed_only"]
    for subset_label, subset_key in [("ALL", "all"), ("WELL-KNOWN", "well_known"), ("ULTRA-RARE", "ultra_rare")]:
        print(f"\n--- {subset_label} ---")
        print(f"{'Config':<20} {'n':<5} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'MRR':<8}")
        print("-" * 57)
        for cn in display:
            if cn not in all_results:
                continue
            m = all_results[cn].get("metrics", {}).get(subset_key, {})
            if not m:
                continue
            n = m.get("n", "?")
            print(f"{cn:<20} {n:<5} {m.get('top1',0):<8.1%} {m.get('top3',0):<8.1%} {m.get('top5',0):<8.1%} {m.get('mrr',0):<8.3f}")


if __name__ == "__main__":
    main()
