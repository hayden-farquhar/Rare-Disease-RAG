"""Assemble retrieved chunks into structured context for LLM generation."""

from typing import Optional
from collections import defaultdict

from src.retrieve.retriever import RetrievedChunk


def assemble_context(
    chunks: list[RetrievedChunk],
    max_tokens: int = 6000,
    chars_per_token: float = 4.0,
) -> str:
    """Assemble retrieved chunks into structured context with source attribution.

    Groups chunks by candidate disease, includes best evidence from each source.

    Args:
        chunks: Ranked list of retrieved chunks.
        max_tokens: Approximate token budget for the context.
        chars_per_token: Approximate characters per token.
    """
    max_chars = int(max_tokens * chars_per_token)

    # Group by disease
    disease_groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        disease_groups[chunk.disease_name].append(chunk)

    # Rank diseases by best chunk score
    disease_order = sorted(
        disease_groups.keys(),
        key=lambda d: max(c.score for c in disease_groups[d]),
        reverse=True,
    )

    # Build context string
    parts = ["## Retrieved Evidence\n"]
    total_chars = len(parts[0])

    for disease_name in disease_order:
        disease_chunks = disease_groups[disease_name]

        # Deduplicate by chunk_type (keep highest scored)
        best_by_type: dict[str, RetrievedChunk] = {}
        for chunk in sorted(disease_chunks, key=lambda c: c.score, reverse=True):
            if chunk.chunk_type not in best_by_type:
                best_by_type[chunk.chunk_type] = chunk

        # Build disease section
        section = f"\n### Candidate: {disease_name}\n"
        for chunk_type in ["overview", "phenotypes", "genetics", "case_presentation", "case_diagnosis"]:
            if chunk_type not in best_by_type:
                continue
            chunk = best_by_type[chunk_type]
            source_tag = f"[{chunk.source.upper()}:{chunk.source_id}]"
            section += f"\n**{chunk_type.title()}** {source_tag}\n{chunk.text}\n"

        # Check token budget
        if total_chars + len(section) > max_chars:
            # Try to fit a truncated version
            remaining = max_chars - total_chars
            if remaining > 200:
                section = section[:remaining] + "\n[...truncated]\n"
                parts.append(section)
            break

        parts.append(section)
        total_chars += len(section)

    return "\n".join(parts)
