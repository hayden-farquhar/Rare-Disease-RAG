"""End-to-end test of the RAG diagnostic pipeline.

Tests retrieval + context assembly. If ANTHROPIC_API_KEY is set, also tests
LLM generation (both RAG and no-RAG baselines).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.index.vector_store import VectorStore
from src.index.embedder import get_embedder
from src.retrieve.retriever import MultiSourceRetriever
from src.retrieve.query_processor import process_query
from src.retrieve.context_assembler import assemble_context
from src.generate.generator import generate_diagnosis, generate_diagnosis_no_rag
from src.ingest.hpo_mapper import HPOOntology
from src.evaluate.metrics import disease_match

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


def main():
    print("=" * 70)
    print("End-to-End RAG Diagnostic Pipeline Test")
    print(f"LLM generation: {'ENABLED' if HAS_API_KEY else 'DISABLED (no ANTHROPIC_API_KEY)'}")
    print("=" * 70)

    # Load components
    print("\n1. Loading components...")
    stores = {}
    index_dir = Path(__file__).resolve().parents[1] / "data" / "processed" / "indices"
    for source in ["orphanet", "pubmed"]:
        source_dir = index_dir / source
        if source_dir.exists():
            stores[source] = VectorStore.load(source, source_dir)
    embedder = get_embedder("all-MiniLM-L6-v2")
    retriever = MultiSourceRetriever(stores=stores, embedder=embedder)
    print(f"   Loaded {len(stores)} indices: {list(stores.keys())}")

    hpo = HPOOntology()
    hpo_index = Path(__file__).resolve().parents[1] / "data" / "hpo" / "hpo_index.json"
    if hpo_index.exists():
        hpo.load_index(hpo_index)
    else:
        hpo = None

    client = None
    if HAS_API_KEY:
        import anthropic
        client = anthropic.Anthropic()

    # Load test cases
    test_path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "test_cases.json"
    with open(test_path) as f:
        test_cases = json.load(f)

    results = []

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"Case: {case['title']}")
        print(f"Target: {case['final_diagnosis']}")
        print(f"{'='*70}")

        vignette = case["clinical_vignette"]
        target = case["final_diagnosis"]
        aliases = case.get("aliases", [])

        # --- Step 1: Query processing ---
        print("\n  Step 1: Query processing (fallback mode)...")
        processed = process_query(
            vignette, hpo_ontology=hpo, use_llm=False
        )
        # Also manually extract key features for multi-query
        key_features = case.get("key_discriminating_features", [])
        if key_features:
            processed.extracted_phenotypes = key_features
            # Generate diverse queries from known features
            queries = [
                " ".join(key_features),
                vignette[:500],
            ]
            # Pairs of distinctive features
            for i in range(0, min(len(key_features), 4), 2):
                if i + 1 < len(key_features):
                    queries.append(f"{key_features[i]} {key_features[i+1]} rare disease")
            processed.retrieval_queries = queries

        print(f"    Phenotypes: {processed.extracted_phenotypes}")
        print(f"    Queries: {len(processed.retrieval_queries)}")
        for i, q in enumerate(processed.retrieval_queries):
            print(f"      Q{i+1}: {q[:80]}...")

        # --- Step 2: HPO mapping ---
        if hpo and processed.extracted_phenotypes:
            print("\n  Step 2: HPO mapping...")
            for pheno in processed.extracted_phenotypes[:5]:
                matches = hpo.search_by_name(pheno, max_results=1)
                if matches:
                    hid, hname, hscore = matches[0]
                    print(f"    '{pheno}' -> {hid} {hname} (score={hscore:.2f})")

        # --- Step 3: Multi-query retrieval ---
        print("\n  Step 3: Multi-query retrieval...")
        retrieved = retriever.retrieve(
            queries=processed.retrieval_queries, top_k=20
        )
        print(f"    Retrieved {len(retrieved)} chunks")

        # Unique diseases in results
        seen = set()
        retrieval_rank = None
        print("    Top retrieved diseases:")
        disease_rank = 0
        for chunk in retrieved:
            if chunk.disease_name not in seen:
                seen.add(chunk.disease_name)
                disease_rank += 1
                is_match = disease_match(chunk.disease_name, target, aliases)
                mark = " <<<" if is_match else ""
                if is_match and retrieval_rank is None:
                    retrieval_rank = disease_rank
                if disease_rank <= 10:
                    print(f"      {disease_rank}. {chunk.disease_name} (score={chunk.score:.4f}){mark}")

        print(f"    Target disease retrieval rank: {retrieval_rank or 'NOT FOUND'}")

        # --- Step 4: Context assembly ---
        print("\n  Step 4: Context assembly...")
        context = assemble_context(retrieved, max_tokens=6000)
        print(f"    Context: {len(context)} chars (~{len(context)//4} tokens)")

        # Show first disease in context
        lines = context.split("\n")
        for line in lines[:15]:
            if line.strip():
                print(f"    | {line[:100]}")

        # --- Step 5: LLM generation (if API key available) ---
        rag_rank = None
        norag_rank = None

        if HAS_API_KEY:
            print("\n  Step 5a: LLM generation (RAG)...")
            rag_output = generate_diagnosis(
                vignette=vignette, context=context, client=client
            )
            print(f"    Differential ({len(rag_output.differential_diagnosis)} candidates):")
            for dx in rag_output.differential_diagnosis:
                is_match = disease_match(dx.disease_name, target, aliases)
                mark = " <<<" if is_match else ""
                if is_match and rag_rank is None:
                    rag_rank = dx.rank
                print(f"      #{dx.rank}: {dx.disease_name} [{dx.confidence}]{mark}")

            print("\n  Step 5b: LLM generation (no-RAG baseline)...")
            norag_output = generate_diagnosis_no_rag(
                vignette=vignette, client=client
            )
            print(f"    Differential ({len(norag_output.differential_diagnosis)} candidates):")
            for dx in norag_output.differential_diagnosis:
                is_match = disease_match(dx.disease_name, target, aliases)
                mark = " <<<" if is_match else ""
                if is_match and norag_rank is None:
                    norag_rank = dx.rank
                print(f"      #{dx.rank}: {dx.disease_name} [{dx.confidence}]{mark}")

        results.append({
            "case": case["title"],
            "target": target,
            "retrieval_rank": retrieval_rank,
            "rag_rank": rag_rank,
            "norag_rank": norag_rank,
            "n_queries": len(processed.retrieval_queries),
            "n_retrieved": len(retrieved),
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    header = f"{'Case':<35} {'Retrieval':<12}"
    if HAS_API_KEY:
        header += f"{'RAG Rank':<12} {'No-RAG':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        ret_rank = str(r["retrieval_rank"]) if r["retrieval_rank"] else "NOT FOUND"
        line = f"{r['case']:<35} {ret_rank:<12}"
        if HAS_API_KEY:
            rag = str(r["rag_rank"]) if r["rag_rank"] else "NOT FOUND"
            norag = str(r["norag_rank"]) if r["norag_rank"] else "NOT FOUND"
            line += f"{rag:<12} {norag:<12}"
        print(line)

    # Save results
    out_path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "e2e_test_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
