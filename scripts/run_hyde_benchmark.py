"""Benchmark HyDE retrieval on existing 55 cases.

Compares: optimised system (no reranker) with and without HyDE.
"""

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
from src.generate.generator import generate_diagnosis
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match

import numpy as np

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


def get_hyde_processed_query(vignette, hpo, client, hyde_cache):
    """Get processed query with HyDE, using cache."""
    from src.retrieve.query_processor import ProcessedQuery
    key = vignette[:200]

    if key in hyde_cache:
        return ProcessedQuery(**hyde_cache[key])

    processed = process_query(vignette, hpo_ontology=hpo, use_llm=True,
                              client=client, use_hyde=True)

    # Disease-name augmentation query
    if processed.extracted_phenotypes:
        processed.retrieval_queries.append(
            "rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5])
        )

    hyde_cache[key] = processed.model_dump()
    save_hyde_cache(hyde_cache)
    return processed


def retrieval_recall(retrieved, target, aliases, k=20):
    for chunk in retrieved[:k]:
        if disease_match(chunk.disease_name, target, aliases):
            return True
    return False


def main():
    print("=" * 70)
    print("HyDE Retrieval Benchmark")
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

    # Optimised retriever: no reranker
    retriever = MultiSourceRetriever(
        stores=stores, embedder=embedder, bm25_stores=bm25_stores,
        phenotype_scorer=phenotype_scorer, phenotype_weight=1.0,
        disease_chunks_map=disease_chunks_map,
    )

    import anthropic
    client = anthropic.Anthropic()

    hyde_cache = load_hyde_cache()

    # Load benchmark
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = BENCHMARK_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    print(f"Loaded {len(all_cases)} benchmark cases")

    results = []
    for i, case in enumerate(all_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        rarity = case.get("rarity", "unknown")
        vignette = case["clinical_vignette"]

        # Process with HyDE
        processed = get_hyde_processed_query(vignette, hpo, client, hyde_cache)
        queries = processed.retrieval_queries or [vignette[:500]]

        query_hpo_ids = []
        if processed.hpo_candidates:
            query_hpo_ids = [h["hpo_id"] for h in processed.hpo_candidates if h.get("score", 0) > 0.5]

        # Retrieve with HyDE document
        retrieved = retriever.retrieve(
            queries=queries, top_k=50, per_query_k=25,
            query_hpo_ids=query_hpo_ids,
            hyde_document=processed.hyde_document,
        )

        # Generate diagnosis (no reranker)
        context = assemble_context(retrieved, max_tokens=6000)
        output = generate_diagnosis(vignette=vignette, context=context, client=client)

        preds = [dx.disease_name for dx in output.differential_diagnosis]
        rank = None
        for j, p in enumerate(preds):
            if disease_match(p, target, aliases):
                rank = j + 1
                break

        ret_recall = retrieval_recall(retrieved, target, aliases, k=20)

        results.append({
            "case_id": case_id,
            "target": target,
            "rarity": rarity,
            "rank": rank,
            "retrieval_recall_20": ret_recall,
            "hyde_preview": processed.hyde_document[:100] if processed.hyde_document else "",
        })

        rank_str = str(rank) if rank else "MISS"
        ret_str = "Y" if ret_recall else "N"
        print(f"  [{i+1}/{len(all_cases)}] {case_id}: rank={rank_str} ret={ret_str}")

        time.sleep(0.5)

    # Metrics
    print(f"\n{'='*70}")
    print("RESULTS — HyDE + Optimised (no reranker)")
    print(f"{'='*70}")

    for subset_label, filter_fn in [
        ("ALL", lambda r: True),
        ("WELL-KNOWN", lambda r: r["rarity"] != "ultra-rare"),
        ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare"),
    ]:
        subset = [r for r in results if filter_fn(r)]
        if not subset:
            continue
        n = len(subset)
        t1 = sum(1 for r in subset if r["rank"] is not None and r["rank"] <= 1) / n
        t3 = sum(1 for r in subset if r["rank"] is not None and r["rank"] <= 3) / n
        t5 = sum(1 for r in subset if r["rank"] is not None and r["rank"] <= 5) / n
        mrr = sum(1.0/r["rank"] for r in subset if r["rank"] is not None) / n
        ret = sum(1 for r in subset if r["retrieval_recall_20"]) / n
        print(f"\n{subset_label} (n={n}):")
        print(f"  Top-1: {t1:.1%}  Top-3: {t3:.1%}  Top-5: {t5:.1%}  MRR: {mrr:.3f}  Ret@20: {ret:.1%}")

    # Save
    out_path = BENCHMARK_DIR / "hyde_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_cases": len(results)}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
