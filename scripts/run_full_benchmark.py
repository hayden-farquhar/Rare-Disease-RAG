"""Run full RAG vs no-RAG benchmark with LLM query expansion and reranking."""

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
from src.generate.generator import generate_diagnosis, generate_diagnosis_no_rag
from src.ingest.orphanet_ingest import load_diseases
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match, top_k_accuracy, mean_reciprocal_rank


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranking")
    parser.add_argument("--no-llm-queries", action="store_true", help="Use fallback queries instead of LLM")
    parser.add_argument("--no-bm25", action="store_true", help="Disable BM25 hybrid retrieval")
    parser.add_argument("--no-hpo-rescue", action="store_true", help="Disable HPO phenotype rescue")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--cases", nargs="+", default=["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"])
    args = parser.parse_args()

    use_reranker = not args.no_reranker
    use_llm_queries = not args.no_llm_queries
    use_bm25 = not args.no_bm25
    use_hpo_rescue = not args.no_hpo_rescue
    embedding_model = args.embedding_model

    print("=" * 70)
    print("Full RAG vs No-RAG Benchmark")
    print(f"  Embedding model: {embedding_model}")
    print(f"  LLM query expansion: {'ON' if use_llm_queries else 'OFF'}")
    print(f"  Cross-encoder reranking: {'ON' if use_reranker else 'OFF'}")
    print(f"  BM25 hybrid: {'ON' if use_bm25 else 'OFF'}")
    print(f"  HPO phenotype rescue: {'ON' if use_hpo_rescue else 'OFF'}")
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
            # Load BM25 if available
            if use_bm25 and (source_dir / "bm25_config.json").exists():
                bm25_stores[source] = BM25Store.load(source, store.chunks, source_dir)
    embedder = get_embedder(embedding_model)
    retriever = MultiSourceRetriever(
        stores=stores, embedder=embedder, bm25_stores=bm25_stores
    )

    # HPO ontology
    hpo = HPOOntology()
    hpo_index = Path(__file__).resolve().parents[1] / "data" / "hpo" / "hpo_index.json"
    if hpo_index.exists():
        hpo.load_index(hpo_index)
    else:
        hpo = None

    # Reranker
    reranker = None
    if use_reranker:
        try:
            reranker = CrossEncoderReranker()
        except Exception as e:
            print(f"  Warning: Could not load reranker: {e}")

    # HPO phenotype scorer
    phenotype_scorer = None
    if use_hpo_rescue and hpo:
        print("  Loading HPO phenotype scorer...")
        all_diseases = load_diseases()
        phenotype_scorer = PhenotypeScorer(hpo, all_diseases)
        # Build chunk lookup by disease name for rescue injection
        orphanet_store = stores.get("orphanet")
        disease_chunks_map = {}
        if orphanet_store:
            for chunk in orphanet_store.chunks:
                disease_chunks_map.setdefault(chunk.disease_name, []).append(chunk)
        print(f"  Phenotype scorer ready ({len(all_diseases)} diseases)")

    import anthropic
    client = anthropic.Anthropic()

    # Load all benchmark cases
    benchmark_dir = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
    all_cases = []
    for fname in args.cases:
        fpath = benchmark_dir / fname
        if fpath.exists():
            with open(fpath) as f:
                cases = json.load(f)
            all_cases.extend(cases)
            print(f"  Loaded {len(cases)} cases from {fname}")

    print(f"\nTotal benchmark cases: {len(all_cases)}")
    print(f"Indices: {list(stores.keys())}")

    results = []
    rag_predictions = []
    norag_predictions = []
    ground_truths = []
    all_aliases = []

    for i, case in enumerate(all_cases):
        case_id = case.get("case_id", f"case_{i}")
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])
        vignette = case["clinical_vignette"]
        key_features = case.get("key_discriminating_features", [])
        rarity = case.get("rarity", "unknown")

        print(f"\n[{i+1}/{len(all_cases)}] {case_id}: {target} [{rarity}]")

        # --- Query Processing ---
        if use_llm_queries:
            print("  Query processing (LLM)...")
            processed = process_query(vignette, hpo_ontology=hpo, use_llm=True, client=client)
            queries = processed.retrieval_queries
            if not queries:
                queries = [vignette[:500]]
        else:
            # Fallback queries from key features
            queries = [vignette[:500]]
            if key_features:
                queries.append(" ".join(key_features))
                for j in range(0, min(len(key_features), 4), 2):
                    if j + 1 < len(key_features):
                        queries.append(f"{key_features[j]} {key_features[j+1]} rare disease")

        print(f"  {len(queries)} retrieval queries")

        # --- Retrieval ---
        retrieved = retriever.retrieve(queries=queries, top_k=30)
        print(f"  Retrieved {len(retrieved)} chunks")

        # --- Reranking ---
        if reranker and retrieved:
            retrieved = reranker.rerank(query=vignette, chunks=retrieved, top_k=15)
            print(f"  Reranked to {len(retrieved)} chunks")

        # --- HPO Phenotype Rescue ---
        if phenotype_scorer and hasattr(processed, 'hpo_candidates') and processed.hpo_candidates:
            query_hpo_ids = [h['hpo_id'] for h in processed.hpo_candidates if h.get('score', 0) > 0.5]
            if query_hpo_ids:
                already_retrieved = {chunk.disease_name for chunk in retrieved}
                rescue_candidates = phenotype_scorer.get_rescue_candidates(
                    query_hpo_ids, already_retrieved, top_k=5, min_score=0.2
                )
                if rescue_candidates:
                    # Inject rescued disease chunks
                    for orpha_code, disease_name, score in rescue_candidates:
                        rescue_chunks = disease_chunks_map.get(disease_name, [])
                        for rc in rescue_chunks[:2]:  # Max 2 chunks per rescued disease
                            retrieved.append(RetrievedChunk(
                                chunk_id=rc.chunk_id,
                                text=rc.text,
                                source=rc.source,
                                source_id=rc.source_id,
                                disease_name=rc.disease_name,
                                chunk_type=rc.chunk_type,
                                score=score * 0.5,  # Lower score than retrieved
                                hpo_terms=rc.hpo_terms,
                            ))
                    print(f"  HPO rescue: injected {len(rescue_candidates)} diseases")

        # Retrieval rank
        seen = set()
        retrieval_rank = None
        rank_counter = 0
        for chunk in retrieved:
            if chunk.disease_name not in seen:
                seen.add(chunk.disease_name)
                rank_counter += 1
                if disease_match(chunk.disease_name, target, aliases):
                    retrieval_rank = rank_counter
                    break

        # --- Context assembly ---
        context = assemble_context(retrieved, max_tokens=6000)

        # --- RAG generation ---
        print("  Generating RAG diagnosis...")
        rag_output = generate_diagnosis(vignette=vignette, context=context, client=client)
        rag_preds = [dx.disease_name for dx in rag_output.differential_diagnosis]
        rag_rank = None
        for j, dx in enumerate(rag_output.differential_diagnosis):
            if disease_match(dx.disease_name, target, aliases):
                rag_rank = j + 1
                break

        # --- No-RAG baseline ---
        print("  Generating no-RAG baseline...")
        norag_output = generate_diagnosis_no_rag(vignette=vignette, client=client)
        norag_preds = [dx.disease_name for dx in norag_output.differential_diagnosis]
        norag_rank = None
        for j, dx in enumerate(norag_output.differential_diagnosis):
            if disease_match(dx.disease_name, target, aliases):
                norag_rank = j + 1
                break

        ret_str = str(retrieval_rank) if retrieval_rank else "MISS"
        rag_str = str(rag_rank) if rag_rank else "MISS"
        norag_str = str(norag_rank) if norag_rank else "MISS"
        print(f"  Retrieval: {ret_str} | RAG: {rag_str} | No-RAG: {norag_str}")
        if rag_output.differential_diagnosis:
            dx1 = rag_output.differential_diagnosis[0]
            print(f"  RAG #1: {dx1.disease_name} [{dx1.confidence}]")
        if norag_output.differential_diagnosis:
            dx1 = norag_output.differential_diagnosis[0]
            print(f"  No-RAG #1: {dx1.disease_name} [{dx1.confidence}]")

        results.append({
            "case_id": case_id,
            "target": target,
            "rarity": rarity,
            "difficulty": case.get("difficulty_rating", "unknown"),
            "retrieval_rank": retrieval_rank,
            "rag_rank": rag_rank,
            "norag_rank": norag_rank,
            "rag_top3": rag_preds[:3],
            "norag_top3": norag_preds[:3],
        })
        rag_predictions.append(rag_preds)
        norag_predictions.append(norag_preds)
        ground_truths.append(target)
        all_aliases.append(aliases)

        # Brief pause between cases to avoid rate limits
        time.sleep(1)

    # --- Metrics ---
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Case':<25} {'Rarity':<12} {'Retrieval':<11} {'RAG':<8} {'No-RAG':<8}")
    print("-" * 64)
    for r in results:
        ret = str(r["retrieval_rank"]) if r["retrieval_rank"] else "MISS"
        rag = str(r["rag_rank"]) if r["rag_rank"] else "MISS"
        norag = str(r["norag_rank"]) if r["norag_rank"] else "MISS"
        print(f"{r['case_id']:<25} {r['rarity']:<12} {ret:<11} {rag:<8} {norag:<8}")

    # Split metrics by rarity
    for subset_name, filter_fn in [
        ("ALL CASES", lambda r: True),
        ("WELL-KNOWN", lambda r: r["rarity"] in ("rare", "uncommon", "easy")),
        ("ULTRA-RARE", lambda r: r["rarity"] == "ultra-rare"),
    ]:
        indices = [i for i, r in enumerate(results) if filter_fn(r)]
        if not indices:
            continue
        sub_rag = [rag_predictions[i] for i in indices]
        sub_norag = [norag_predictions[i] for i in indices]
        sub_gt = [ground_truths[i] for i in indices]
        sub_aliases = [all_aliases[i] for i in indices]

        print(f"\n{'='*70}")
        print(f"METRICS — {subset_name} (n={len(indices)})")
        print(f"{'='*70}")
        print(f"{'Metric':<25} {'RAG':<12} {'No-RAG':<12}")
        print("-" * 49)
        for k in [1, 3, 5]:
            rag_acc = top_k_accuracy(sub_rag, sub_gt, sub_aliases, k=k)
            norag_acc = top_k_accuracy(sub_norag, sub_gt, sub_aliases, k=k)
            print(f"Top-{k} accuracy          {rag_acc:.1%}        {norag_acc:.1%}")
        rag_mrr = mean_reciprocal_rank(sub_rag, sub_gt, sub_aliases)
        norag_mrr = mean_reciprocal_rank(sub_norag, sub_gt, sub_aliases)
        print(f"MRR                      {rag_mrr:.3f}       {norag_mrr:.3f}")

    # Save
    config = {
        "llm_queries": use_llm_queries,
        "reranker": use_reranker,
        "n_cases": len(all_cases),
    }
    out_path = benchmark_dir / "full_benchmark_results.json"
    all_metrics = {}
    for subset_name, filter_fn in [("all", lambda r: True), ("well_known", lambda r: r["rarity"] != "ultra-rare"), ("ultra_rare", lambda r: r["rarity"] == "ultra-rare")]:
        indices = [i for i, r in enumerate(results) if filter_fn(r)]
        if not indices:
            continue
        sub_rag = [rag_predictions[i] for i in indices]
        sub_norag = [norag_predictions[i] for i in indices]
        sub_gt = [ground_truths[i] for i in indices]
        sub_aliases = [all_aliases[i] for i in indices]
        all_metrics[subset_name] = {
            "rag": {f"top{k}": top_k_accuracy(sub_rag, sub_gt, sub_aliases, k=k) for k in [1,3,5]},
            "norag": {f"top{k}": top_k_accuracy(sub_norag, sub_gt, sub_aliases, k=k) for k in [1,3,5]},
        }
        all_metrics[subset_name]["rag"]["mrr"] = mean_reciprocal_rank(sub_rag, sub_gt, sub_aliases)
        all_metrics[subset_name]["norag"]["mrr"] = mean_reciprocal_rank(sub_norag, sub_gt, sub_aliases)

    with open(out_path, "w") as f:
        json.dump({"config": config, "results": results, "metrics": all_metrics}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
