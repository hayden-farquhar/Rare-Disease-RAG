"""Chunk PubMed case reports for embedding and retrieval."""

from src.index.chunker import Chunk
from src.ingest.pubmed_cases import CaseReport


def chunk_case_report(case: CaseReport) -> list[Chunk]:
    """Create chunks from a PubMed case report.

    Each case becomes 1-2 chunks:
    1. Case presentation (title + abstract)
    2. MeSH-enriched version (title + MeSH terms as context)
    """
    chunks = []

    if not case.abstract:
        return chunks

    # Chunk 1: Full case (title + abstract)
    text_parts = [f"Case Report: {case.title}"]
    if case.journal:
        text_parts.append(f"Journal: {case.journal} ({case.year})")
    text_parts.append(f"\n{case.abstract}")

    if case.mesh_terms:
        mesh_str = ", ".join(case.mesh_terms)
        text_parts.append(f"\nMeSH terms: {mesh_str}")

    chunks.append(Chunk(
        chunk_id=f"PMID:{case.pmid}_case",
        text="\n".join(text_parts),
        source="pubmed",
        source_id=f"PMID:{case.pmid}",
        disease_name=_extract_disease_from_mesh(case.mesh_terms) or case.title,
        chunk_type="case_presentation",
        metadata={"pmid": case.pmid, "year": case.year, "journal": case.journal},
    ))

    return chunks


def _extract_disease_from_mesh(mesh_terms: list[str]) -> str:
    """Try to extract a disease name from MeSH terms.

    Skip generic terms like 'Humans', 'Male', 'Female', etc.
    """
    skip = {
        "Humans", "Male", "Female", "Adult", "Child", "Infant",
        "Infant, Newborn", "Middle Aged", "Aged", "Young Adult",
        "Adolescent", "Child, Preschool", "Fatal Outcome",
        "Pregnancy", "Gestational Age", "Treatment Outcome",
    }
    for term in mesh_terms:
        if term not in skip and not term.startswith("diagnosis"):
            return term
    return ""


def chunk_all_cases(cases: list[CaseReport]) -> list[Chunk]:
    """Chunk all case reports."""
    all_chunks = []
    for case in cases:
        all_chunks.extend(chunk_case_report(case))
    return all_chunks
