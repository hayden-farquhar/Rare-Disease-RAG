"""Compare all diagnostic approaches: Standard RAG, No-RAG, Hypothesis-Validate, Ensemble."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.retrieve.retriever import MultiSourceRetriever, CrossEncoderReranker, RetrievedChunk
from src.retrieve.bm25_store import BM25Store
from src.retrieve.query_processor import process_query
from src.retrieve.context_assembler import assemble_context
from src.retrieve.phenotype_scorer import PhenotypeScorer
from src.generate.generator import generate_diagnosis, generate_diagnosis_no_rag
from src.generate.hypothesis_validate import (
    diagnose_hypothesis_validate, diagnose_ensemble,
)
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match, top_k_accuracy, mean_reciprocal_rank
from src.index.chunker import Chunk


def main():
    print("=" * 70)
    print("Comprehensive Diagnostic Approach Comparison")
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

    # HPO
    hpo = HPOOntology()
    hpo_index = Path(__file__).resolve().parents[1] / "data" / "hpo" / "hpo_index.json"
    if hpo_index.exists():
        hpo.load_index(hpo_index)

    # Phenotype scorer + disease chunk map
    all_diseases = load_diseases()
    phenotype_scorer = PhenotypeScorer(hpo, all_diseases)
    disease_chunks_map = {}
    orphanet_store = stores.get("orphanet")
    if orphanet_store:
        for chunk in orphanet_store.chunks:
            disease_chunks_map.setdefault(chunk.disease_name, []).append(chunk)

    # Retriever with all channels
    retriever = MultiSourceRetriever(
        stores=stores, embedder=embedder, bm25_stores=bm25_stores,
        phenotype_scorer=phenotype_scorer, phenotype_weight=1.0,
        disease_chunks_map=disease_chunks_map,
    )

    # Reranker
    reranker = CrossEncoderReranker()

    import anthropic
    client = anthropic.Anthropic()

    # Load benchmark
    benchmark_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = benchmark_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    print(f"Loaded {len(all_cases)} benchmark cases")

    # Results storage
    approaches = ["rag", "norag", "hyp_validate", "ensemble"]
    predictions = {a: [] for a in approaches}
    ground_truths = []
    all_aliases = []
    results = []

    for i, case in enumerate(all_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        vignette = case["clinical_vignette"]
        rarity = case.get("rarity", "unknown")

        print(f"\n[{i+1}/{len(all_cases)}] {case_id}: {target} [{rarity}]")

        # === Standard RAG (with all improvements) ===
        processed = process_query(vignette, hpo_ontology=hpo, use_llm=True, client=client)
        queries = processed.retrieval_queries or [vignette[:500]]

        # Query augmentation: add disease-name-oriented query
        if processed.extracted_phenotypes:
            queries.append("rare genetic syndrome with " + ", ".join(processed.extracted_phenotypes[:5]))

        # Extract HPO IDs for phenotype-based retrieval channel
        query_hpo_ids = []
        if processed.hpo_candidates:
            query_hpo_ids = [h['hpo_id'] for h in processed.hpo_candidates if h.get('score', 0) > 0.5]

        # Retrieve with increased depth + HPO channel
        retrieved = retriever.retrieve(
            queries=queries, top_k=50, per_query_k=25,
            query_hpo_ids=query_hpo_ids,
        )
        if reranker:
            retrieved = reranker.rerank(query=vignette, chunks=retrieved, top_k=20)

        context = assemble_context(retrieved, max_tokens=6000)
        rag_output = generate_diagnosis(vignette=vignette, context=context, client=client)
        rag_preds = [dx.disease_name for dx in rag_output.differential_diagnosis]

        # === No-RAG ===
        norag_output = generate_diagnosis_no_rag(vignette=vignette, client=client)
        norag_preds = [dx.disease_name for dx in norag_output.differential_diagnosis]

        # === Hypothesis-Validate ===
        hv_output, hv_candidates = diagnose_hypothesis_validate(
            vignette=vignette, client=client,
            bm25_stores=bm25_stores, vector_stores=stores,
        )
        hv_preds = [dx.disease_name for dx in hv_output.differential_diagnosis]

        # === Ensemble (RAG + No-RAG) ===
        ens_output = diagnose_ensemble(vignette, rag_output, norag_output)
        ens_preds = [dx.disease_name for dx in ens_output.differential_diagnosis]

        # Record results
        result = {"case_id": case_id, "target": target, "rarity": rarity}
        for approach, preds in [("rag", rag_preds), ("norag", norag_preds),
                                 ("hyp_validate", hv_preds), ("ensemble", ens_preds)]:
            rank = None
            for j, p in enumerate(preds):
                if disease_match(p, target, aliases):
                    rank = j + 1
                    break
            result[f"{approach}_rank"] = rank
            predictions[approach].append(preds)

        ground_truths.append(target)
        all_aliases.append(aliases)
        results.append(result)

        # Print summary for this case
        ranks = {a: str(result[f"{a}_rank"]) if result[f"{a}_rank"] else "MISS" for a in approaches}
        print(f"  RAG:{ranks['rag']} | NoRAG:{ranks['norag']} | HypVal:{ranks['hyp_validate']} | Ensemble:{ranks['ensemble']}")

        time.sleep(1)

    # === METRICS ===
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Per-case table
    print(f"\n{'Case':<22} {'Rarity':<12} {'RAG':<7} {'NoRAG':<7} {'HypVal':<7} {'Ensem':<7}")
    print("-" * 62)
    for r in results:
        vals = [str(r[f"{a}_rank"]) if r[f"{a}_rank"] else "MISS" for a in approaches]
        print(f"{r['case_id']:<22} {r['rarity']:<12} {vals[0]:<7} {vals[1]:<7} {vals[2]:<7} {vals[3]:<7}")

    # Metrics by subset
    for subset_name, filter_fn in [
        ("ALL CASES", lambda r: True),
        ("WELL-KNOWN", lambda r: r["rarity"] != "ultra-rare"),
        ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare"),
    ]:
        indices = [i for i, r in enumerate(results) if filter_fn(r)]
        if not indices:
            continue

        print(f"\n{'='*70}")
        print(f"METRICS — {subset_name} (n={len(indices)})")
        print(f"{'='*70}")
        print(f"{'Metric':<20} {'RAG':<10} {'NoRAG':<10} {'HypVal':<10} {'Ensemble':<10}")
        print("-" * 60)

        for k in [1, 3, 5]:
            vals = []
            for approach in approaches:
                sub_preds = [predictions[approach][i] for i in indices]
                sub_gt = [ground_truths[i] for i in indices]
                sub_al = [all_aliases[i] for i in indices]
                acc = top_k_accuracy(sub_preds, sub_gt, sub_al, k=k)
                vals.append(f"{acc:.1%}")
            print(f"Top-{k}               {vals[0]:<10} {vals[1]:<10} {vals[2]:<10} {vals[3]:<10}")

        vals = []
        for approach in approaches:
            sub_preds = [predictions[approach][i] for i in indices]
            sub_gt = [ground_truths[i] for i in indices]
            sub_al = [all_aliases[i] for i in indices]
            mrr = mean_reciprocal_rank(sub_preds, sub_gt, sub_al)
            vals.append(f"{mrr:.3f}")
        print(f"MRR                 {vals[0]:<10} {vals[1]:<10} {vals[2]:<10} {vals[3]:<10}")

    # Save
    out_path = benchmark_dir / "all_approaches_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "n_cases": len(all_cases)}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
