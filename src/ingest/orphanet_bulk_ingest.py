"""Bulk ingest Orphanet data from downloaded XML product files.

Parses product4 (phenotypes), product6 (genes), product9_ages (natural history),
and product1 (cross-references) XML files to build comprehensive disease records.

Much faster than per-disease API calls for large-scale ingestion.
"""

import json
from pathlib import Path
from typing import Optional

from lxml import etree

from src.ingest.orphanet_ingest import (
    RareDisease, HPOAssociation, GeneAssociation, _save_diseases,
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "orphanet"


def parse_product4_phenotypes(path: Optional[Path] = None) -> dict[int, list[HPOAssociation]]:
    """Parse product4 XML: disease-HPO phenotype associations."""
    path = path or RAW_DIR / "en_product4.xml"
    print(f"Parsing phenotypes from {path.name}...")

    disease_hpo: dict[int, list[HPOAssociation]] = {}

    context = etree.iterparse(str(path), events=("end",), tag="Disorder")
    for event, disorder in context:
        orpha_el = disorder.find(".//OrphaCode")
        if orpha_el is None or orpha_el.text is None:
            disorder.clear()
            continue
        orpha_code = int(orpha_el.text)

        associations = []
        for assoc in disorder.findall(".//HPODisorderAssociation"):
            hpo_el = assoc.find(".//HPO")
            if hpo_el is None:
                continue
            hpo_id_el = hpo_el.find("HPOId")
            hpo_term_el = hpo_el.find("HPOTerm")
            freq_el = assoc.find(".//HPOFrequency/Name")

            if hpo_id_el is not None and hpo_id_el.text:
                associations.append(HPOAssociation(
                    hpo_id=hpo_id_el.text,
                    hpo_term=hpo_term_el.text if hpo_term_el is not None else "",
                    frequency=freq_el.text if freq_el is not None else None,
                ))

        if associations:
            disease_hpo[orpha_code] = associations

        disorder.clear()

    print(f"  Parsed phenotypes for {len(disease_hpo)} diseases")
    return disease_hpo


def parse_product6_genes(path: Optional[Path] = None) -> dict[int, list[GeneAssociation]]:
    """Parse product6 XML: disease-gene associations."""
    path = path or RAW_DIR / "en_product6.xml"
    print(f"Parsing genes from {path.name}...")

    disease_genes: dict[int, list[GeneAssociation]] = {}

    context = etree.iterparse(str(path), events=("end",), tag="Disorder")
    for event, disorder in context:
        orpha_el = disorder.find(".//OrphaCode")
        if orpha_el is None or orpha_el.text is None:
            disorder.clear()
            continue
        orpha_code = int(orpha_el.text)

        genes = []
        for assoc in disorder.findall(".//DisorderGeneAssociation"):
            gene_el = assoc.find(".//Gene")
            if gene_el is None:
                continue
            symbol_el = gene_el.find("Symbol")
            name_el = gene_el.find("Name")
            type_el = assoc.find(".//DisorderGeneAssociationType/Name")
            locus_el = gene_el.find(".//Locus/GeneLocus")

            if symbol_el is not None and symbol_el.text:
                genes.append(GeneAssociation(
                    gene_symbol=symbol_el.text,
                    gene_name=name_el.text if name_el is not None else "",
                    gene_type=type_el.text if type_el is not None else None,
                    locus=locus_el.text if locus_el is not None else None,
                ))

        if genes:
            disease_genes[orpha_code] = genes

        disorder.clear()

    print(f"  Parsed genes for {len(disease_genes)} diseases")
    return disease_genes


def parse_product9_natural_history(path: Optional[Path] = None) -> dict[int, dict]:
    """Parse product9_ages XML: age of onset, inheritance."""
    path = path or RAW_DIR / "en_product9_ages.xml"
    print(f"Parsing natural history from {path.name}...")

    disease_history: dict[int, dict] = {}

    context = etree.iterparse(str(path), events=("end",), tag="Disorder")
    for event, disorder in context:
        orpha_el = disorder.find(".//OrphaCode")
        if orpha_el is None or orpha_el.text is None:
            disorder.clear()
            continue
        orpha_code = int(orpha_el.text)

        inheritance = []
        for inh in disorder.findall(".//TypeOfInheritance/Name"):
            if inh.text:
                inheritance.append(inh.text)

        age_of_onset = []
        for age in disorder.findall(".//AverageAgeOfOnset/Name"):
            if age.text:
                age_of_onset.append(age.text)

        if inheritance or age_of_onset:
            disease_history[orpha_code] = {
                "inheritance": inheritance,
                "age_of_onset": age_of_onset,
            }

        disorder.clear()

    print(f"  Parsed natural history for {len(disease_history)} diseases")
    return disease_history


def parse_product1_crossrefs(path: Optional[Path] = None) -> dict[int, dict]:
    """Parse product1 XML: disease names, OMIM/ICD cross-references."""
    path = path or RAW_DIR / "en_product1.xml"
    print(f"Parsing cross-references from {path.name}...")

    disease_info: dict[int, dict] = {}

    context = etree.iterparse(str(path), events=("end",), tag="Disorder")
    for event, disorder in context:
        orpha_el = disorder.find(".//OrphaCode")
        if orpha_el is None or orpha_el.text is None:
            disorder.clear()
            continue
        orpha_code = int(orpha_el.text)

        name_el = disorder.find(".//Name")
        name = name_el.text if name_el is not None else f"ORPHA:{orpha_code}"

        # Synonyms
        synonyms = []
        for syn in disorder.findall(".//SynonymList/Synonym"):
            if syn.text:
                synonyms.append(syn.text)

        # External references (OMIM, ICD-10)
        omim_ids = []
        icd10_codes = []
        for ref in disorder.findall(".//ExternalReference"):
            source_el = ref.find("Source")
            ref_el = ref.find("Reference")
            if source_el is not None and ref_el is not None:
                source = source_el.text or ""
                reference = ref_el.text or ""
                if source == "OMIM" and reference:
                    omim_ids.append(reference)
                elif source == "ICD-10" and reference:
                    icd10_codes.append(reference)

        disease_info[orpha_code] = {
            "name": name,
            "synonyms": synonyms,
            "omim_ids": omim_ids,
            "icd10_codes": icd10_codes,
        }

        disorder.clear()

    print(f"  Parsed cross-references for {len(disease_info)} diseases")
    return disease_info


def build_disease_records(
    min_hpo_count: int = 3,
    output_path: Optional[Path] = None,
) -> list[RareDisease]:
    """Build comprehensive disease records from all XML product files.

    Args:
        min_hpo_count: Minimum HPO associations to include a disease.
        output_path: Where to save the JSON output.
    """
    output_path = output_path or RAW_DIR / "diseases.json"

    # Parse all data sources
    phenotypes = parse_product4_phenotypes()
    genes = parse_product6_genes()
    history = parse_product9_natural_history()
    crossrefs = parse_product1_crossrefs()

    # Build records for diseases that have phenotype data
    diseases = []
    for orpha_code, hpo_list in phenotypes.items():
        if len(hpo_list) < min_hpo_count:
            continue

        info = crossrefs.get(orpha_code, {})
        hist = history.get(orpha_code, {})
        gene_list = genes.get(orpha_code, [])

        name = info.get("name", f"ORPHA:{orpha_code}")

        # Skip disease groups/categories (usually have generic names)
        if name.startswith("OBSOLETE:") or name.startswith("NON RARE"):
            continue

        disease = RareDisease(
            orpha_code=orpha_code,
            name=name,
            synonyms=info.get("synonyms", []),
            omim_ids=info.get("omim_ids", []),
            icd10_codes=info.get("icd10_codes", []),
            inheritance=hist.get("inheritance", []),
            age_of_onset=hist.get("age_of_onset", []),
            hpo_associations=hpo_list,
            genes=gene_list,
        )
        diseases.append(disease)

    # Sort by number of HPO associations (richest data first)
    diseases.sort(key=lambda d: len(d.hpo_associations), reverse=True)

    _save_diseases(diseases, output_path)
    print(f"\nBuilt {len(diseases)} disease records (min {min_hpo_count} HPO associations)")
    print(f"  Total HPO associations: {sum(len(d.hpo_associations) for d in diseases)}")
    print(f"  Diseases with genes: {sum(1 for d in diseases if d.genes)}")
    print(f"  Diseases with inheritance: {sum(1 for d in diseases if d.inheritance)}")
    print(f"  Diseases with OMIM IDs: {sum(1 for d in diseases if d.omim_ids)}")
    print(f"Saved to {output_path}")

    return diseases


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Orphanet disease records from XML files")
    parser.add_argument("--min-hpo", type=int, default=3,
                        help="Minimum HPO associations per disease (default: 3)")
    args = parser.parse_args()
    build_disease_records(min_hpo_count=args.min_hpo)
