"""Document chunking for rare disease knowledge base.

Creates text chunks from disease records, optimised for embedding and retrieval.
Each chunk includes metadata for source attribution.
"""

from dataclasses import dataclass, field
from typing import Optional

from src.ingest.orphanet_ingest import RareDisease


@dataclass
class Chunk:
    """A text chunk with metadata for retrieval."""
    chunk_id: str
    text: str
    source: str  # "orphanet", "omim", "pubmed"
    source_id: str  # ORPHA code, OMIM ID, PMID
    disease_name: str
    chunk_type: str  # "overview", "phenotypes", "genetics", "case_presentation", "case_diagnosis"
    hpo_terms: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "source_id": self.source_id,
            "disease_name": self.disease_name,
            "chunk_type": self.chunk_type,
            "hpo_terms": self.hpo_terms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def chunk_orphanet_disease(disease: RareDisease) -> list[Chunk]:
    """Create text chunks from an Orphanet disease record.

    Produces 2-3 chunks per disease:
    1. Overview: name, synonyms, prevalence, inheritance, genes
    2. Phenotypes: HPO-annotated clinical features grouped by frequency
    3. Genetics: gene associations and molecular details (if genes present)
    """
    chunks = []
    orpha_id = f"ORPHA:{disease.orpha_code}"

    # Chunk 1: Disease overview
    overview_parts = [f"Disease: {disease.name}"]
    if disease.synonyms:
        overview_parts.append(f"Synonyms: {', '.join(disease.synonyms)}")
    overview_parts.append(f"ORPHAcode: {disease.orpha_code}")
    if disease.omim_ids:
        overview_parts.append(f"OMIM: {', '.join(disease.omim_ids)}")
    if disease.prevalence:
        overview_parts.append(f"Prevalence: {disease.prevalence}")
    if disease.inheritance:
        overview_parts.append(f"Inheritance: {', '.join(disease.inheritance)}")
    if disease.age_of_onset:
        overview_parts.append(f"Age of onset: {', '.join(disease.age_of_onset)}")
    if disease.genes:
        gene_str = ", ".join(f"{g.gene_symbol} ({g.gene_name})" for g in disease.genes)
        overview_parts.append(f"Associated genes: {gene_str}")
    if disease.description:
        overview_parts.append(f"\nClinical description: {disease.description}")

    chunks.append(Chunk(
        chunk_id=f"{orpha_id}_overview",
        text="\n".join(overview_parts),
        source="orphanet",
        source_id=orpha_id,
        disease_name=disease.name,
        chunk_type="overview",
        hpo_terms=[h.hpo_id for h in disease.hpo_associations],
        metadata={
            "orpha_code": disease.orpha_code,
            "omim_ids": disease.omim_ids,
        },
    ))

    # Chunk 2: Phenotypes (grouped by frequency)
    if disease.hpo_associations:
        freq_groups: dict[str, list[str]] = {}
        for assoc in disease.hpo_associations:
            freq = assoc.frequency or "Unknown frequency"
            label = f"{assoc.hpo_term} ({assoc.hpo_id})"
            freq_groups.setdefault(freq, []).append(label)

        # Order by clinical importance
        freq_order = ["Obligate", "Very frequent", "Frequent", "Occasional", "Very rare", "Excluded"]
        pheno_parts = [f"Clinical features of {disease.name}:"]
        for freq in freq_order:
            if freq in freq_groups:
                pheno_parts.append(f"\n{freq}:")
                for feature in freq_groups[freq]:
                    pheno_parts.append(f"  - {feature}")
                del freq_groups[freq]
        # Any remaining frequencies not in the standard order
        for freq, features in freq_groups.items():
            pheno_parts.append(f"\n{freq}:")
            for feature in features:
                pheno_parts.append(f"  - {feature}")

        chunks.append(Chunk(
            chunk_id=f"{orpha_id}_phenotypes",
            text="\n".join(pheno_parts),
            source="orphanet",
            source_id=orpha_id,
            disease_name=disease.name,
            chunk_type="phenotypes",
            hpo_terms=[h.hpo_id for h in disease.hpo_associations],
        ))

    # Chunk 3: Genetics (if genes present)
    if disease.genes:
        gene_parts = [f"Genetic basis of {disease.name}:"]
        for gene in disease.genes:
            gene_parts.append(f"\nGene: {gene.gene_symbol} ({gene.gene_name})")
            if gene.gene_type:
                gene_parts.append(f"  Association type: {gene.gene_type}")
            if gene.locus:
                gene_parts.append(f"  Locus: {gene.locus}")

        chunks.append(Chunk(
            chunk_id=f"{orpha_id}_genetics",
            text="\n".join(gene_parts),
            source="orphanet",
            source_id=orpha_id,
            disease_name=disease.name,
            chunk_type="genetics",
            hpo_terms=[],
        ))

    return chunks


def chunk_orphanet_disease_with_narrative(
    disease: RareDisease, narrative: str
) -> list[Chunk]:
    """Create chunks including a narrative clinical description."""
    chunks = chunk_orphanet_disease(disease)
    orpha_id = f"ORPHA:{disease.orpha_code}"

    if narrative:
        chunks.append(Chunk(
            chunk_id=f"{orpha_id}_narrative",
            text=f"Clinical presentation of {disease.name}: {narrative}",
            source="orphanet",
            source_id=orpha_id,
            disease_name=disease.name,
            chunk_type="narrative",
            hpo_terms=[h.hpo_id for h in disease.hpo_associations],
        ))

    return chunks


def chunk_all_diseases(
    diseases: list[RareDisease],
    narratives: dict[int, str] | None = None,
) -> list[Chunk]:
    """Chunk all disease records, optionally with narratives."""
    all_chunks = []
    narratives = narratives or {}
    for disease in diseases:
        narrative = narratives.get(disease.orpha_code, "")
        if narrative:
            all_chunks.extend(chunk_orphanet_disease_with_narrative(disease, narrative))
        else:
            all_chunks.extend(chunk_orphanet_disease(disease))
    return all_chunks
