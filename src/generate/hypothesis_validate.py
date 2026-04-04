"""Hypothesis-validate diagnostic approach.

Instead of: vignette -> embed -> search -> generate
Does: vignette -> LLM hypothesises candidates -> look up each in KB -> generate with evidence

This bypasses the embedding quality bottleneck by using the LLM's
medical knowledge to generate candidate disease names, then using
BM25 keyword matching to retrieve their knowledge base entries.
"""

import json
import time
from typing import Optional

from src.retrieve.bm25_store import BM25Store
from src.index.vector_store import VectorStore
from src.retrieve.context_assembler import assemble_context
from src.retrieve.retriever import RetrievedChunk
from src.generate.generator import generate_diagnosis, DiagnosticOutput, _call_with_retry


def generate_candidates(vignette: str, client, model: str = "claude-sonnet-4-20250514",
                        n_candidates: int = 15) -> list[str]:
    """Ask the LLM to hypothesise candidate rare diseases from a vignette."""
    prompt = (
        "You are an expert clinical geneticist. Given this clinical vignette, "
        f"list {n_candidates} possible rare disease diagnoses, ordered by likelihood. "
        "Include ultra-rare conditions and genetic syndromes. "
        "Think broadly — consider conditions that match even a subset of the features.\n\n"
        f"Vignette: {vignette}\n\n"
        "Return ONLY a JSON array of disease name strings, e.g. "
        '[\"Disease A\", \"Disease B\", ...]'
    )

    response = _call_with_retry(client, model, 800, 0.3, prompt)
    text = response.content[0].text.strip()

    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def lookup_candidates_in_kb(
    candidates: list[str],
    bm25_stores: dict[str, BM25Store],
    vector_stores: dict[str, VectorStore],
    max_chunks_per_candidate: int = 3,
) -> list[RetrievedChunk]:
    """Look up each candidate disease in the knowledge base via BM25."""
    all_chunks = []
    seen_ids = set()

    for candidate in candidates:
        for source_name, bm25 in bm25_stores.items():
            results = bm25.search(candidate, top_k=max_chunks_per_candidate)
            for r in results:
                if r.chunk.chunk_id not in seen_ids and r.score > 0.05:
                    seen_ids.add(r.chunk.chunk_id)
                    all_chunks.append(RetrievedChunk(
                        chunk_id=r.chunk.chunk_id,
                        text=r.chunk.text,
                        source=r.chunk.source,
                        source_id=r.chunk.source_id,
                        disease_name=r.chunk.disease_name,
                        chunk_type=r.chunk.chunk_type,
                        score=r.score,
                        hpo_terms=r.chunk.hpo_terms,
                    ))

    return all_chunks


def diagnose_hypothesis_validate(
    vignette: str,
    client,
    bm25_stores: dict[str, BM25Store],
    vector_stores: dict[str, VectorStore],
    model: str = "claude-sonnet-4-20250514",
    n_candidates: int = 15,
) -> DiagnosticOutput:
    """Full hypothesis-validate pipeline.

    1. LLM generates candidate disease names
    2. Look up each candidate in knowledge base (BM25)
    3. Assemble evidence context
    4. Generate final diagnosis with evidence
    """
    # Step 1: Generate candidates
    candidates = generate_candidates(vignette, client, model, n_candidates)

    # Step 2: Look up in knowledge base
    retrieved = lookup_candidates_in_kb(candidates, bm25_stores, vector_stores)

    # Step 3: Assemble context
    context = assemble_context(retrieved, max_tokens=6000)

    # Step 4: Generate with evidence
    output = generate_diagnosis(
        vignette=vignette, context=context, client=client, model=model,
    )

    return output, candidates


def diagnose_ensemble(
    vignette: str,
    rag_output: DiagnosticOutput,
    norag_output: DiagnosticOutput,
) -> DiagnosticOutput:
    """Ensemble: merge RAG and no-RAG differentials.

    Diseases appearing in both lists get boosted.
    Final ranking by: appeared_in_both > RAG_rank > noRAG_rank.
    """
    from src.generate.generator import DiagnosisCandidate
    from src.evaluate.metrics import normalize_disease_name

    # Collect all candidates with their sources
    candidates = {}  # normalized name -> {rag_rank, norag_rank, best_entry}

    for dx in rag_output.differential_diagnosis:
        key = normalize_disease_name(dx.disease_name)
        candidates[key] = {
            "rag_rank": dx.rank,
            "norag_rank": None,
            "entry": dx,
            "sources": ["rag"],
        }

    for dx in norag_output.differential_diagnosis:
        key = normalize_disease_name(dx.disease_name)
        if key in candidates:
            candidates[key]["norag_rank"] = dx.rank
            candidates[key]["sources"].append("norag")
        else:
            candidates[key] = {
                "rag_rank": None,
                "norag_rank": dx.rank,
                "entry": dx,
                "sources": ["norag"],
            }

    # Score: both=0 (best), rag_only=1, norag_only=2. Then by best rank.
    def sort_key(item):
        key, info = item
        in_both = len(info["sources"]) == 2
        best_rank = min(
            info["rag_rank"] or 99,
            info["norag_rank"] or 99,
        )
        return (0 if in_both else 1, best_rank)

    sorted_candidates = sorted(candidates.items(), key=sort_key)

    # Build ensemble output
    ensemble = DiagnosticOutput(
        clinical_summary=rag_output.clinical_summary or norag_output.clinical_summary,
        diagnostic_uncertainty="Ensemble of RAG and LLM-only approaches",
    )

    for rank, (key, info) in enumerate(sorted_candidates, 1):
        entry = info["entry"]
        ensemble.differential_diagnosis.append(DiagnosisCandidate(
            rank=rank,
            disease_name=entry.disease_name,
            orpha_code=entry.orpha_code,
            omim_id=entry.omim_id,
            confidence=entry.confidence if len(info["sources"]) == 2 else "medium",
            supporting_features=entry.supporting_features,
            against_features=entry.against_features,
            evidence_sources=info["sources"],
            reasoning=entry.reasoning,
        ))

    return ensemble
