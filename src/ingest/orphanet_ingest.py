"""Ingest rare disease data from the Orphadata REST API.

Fetches disease records with clinical descriptions, phenotype-HPO associations,
gene associations, natural history, and epidemiology. Outputs structured JSON
disease records ready for chunking and embedding.

API docs: https://api.orphadata.com/
"""

import json
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

BASE_URL = "https://api.orphadata.com"
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "orphanet"


class HPOAssociation(BaseModel):
    hpo_id: str
    hpo_term: str
    frequency: Optional[str] = None
    is_diagnostic_criterion: Optional[bool] = None


class GeneAssociation(BaseModel):
    gene_symbol: str
    gene_name: str
    gene_type: Optional[str] = None
    locus: Optional[str] = None


class RareDisease(BaseModel):
    orpha_code: int
    name: str
    synonyms: list[str] = []
    omim_ids: list[str] = []
    icd10_codes: list[str] = []
    prevalence: Optional[str] = None
    inheritance: list[str] = []
    age_of_onset: list[str] = []
    hpo_associations: list[HPOAssociation] = []
    genes: list[GeneAssociation] = []
    description: Optional[str] = None


class OrphanetClient:
    """Client for the Orphadata REST API."""

    def __init__(self, base_url: str = BASE_URL, delay: float = 0.25):
        self.base_url = base_url
        self.delay = delay
        self.client = httpx.Client(timeout=30.0)

    def _get(self, path: str) -> dict | None:
        url = f"{self.base_url}{path}"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            print(f"  HTTP {e.response.status_code} for {url}")
            return None
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
        finally:
            time.sleep(self.delay)

    def _get_results(self, path: str) -> dict | list | None:
        """Fetch and unwrap the data.results from the API response."""
        raw = self._get(path)
        if not raw:
            return None
        return raw.get("data", {}).get("results", None)

    def get_all_orphacodes(self) -> list[dict]:
        """Fetch all ORPHAcodes with preferred terms."""
        results = self._get_results("/rd-cross-referencing/orphacodes")
        if isinstance(results, list):
            return results
        return []

    def get_cross_references(self, orpha_code: int) -> dict | None:
        """Get OMIM, ICD-10/11 cross-references for a disease."""
        return self._get_results(f"/rd-cross-referencing/orphacodes/{orpha_code}")

    def get_phenotypes(self, orpha_code: int) -> list[HPOAssociation]:
        """Get HPO phenotype associations for a disease."""
        results = self._get_results(f"/rd-phenotypes/orphacodes/{orpha_code}")
        if not results:
            return []

        # Structure: results.Disorder.HPODisorderAssociation[]
        disorder = results.get("Disorder", results) if isinstance(results, dict) else {}
        hpo_list = disorder.get("HPODisorderAssociation", [])
        if not isinstance(hpo_list, list):
            return []

        associations = []
        for item in hpo_list:
            hpo = item.get("HPO", {})
            hpo_id = hpo.get("HPOId", "")
            hpo_term = hpo.get("HPOTerm", "")
            freq = item.get("HPOFrequency", None)
            diag = item.get("DiagnosticCriteria", None)
            if hpo_id:
                associations.append(HPOAssociation(
                    hpo_id=hpo_id,
                    hpo_term=hpo_term,
                    frequency=freq,
                    is_diagnostic_criterion=diag is not None and diag != "null",
                ))
        return associations

    def get_genes(self, orpha_code: int) -> list[GeneAssociation]:
        """Get gene associations for a disease."""
        results = self._get_results(f"/rd-associated-genes/orphacodes/{orpha_code}")
        if not results:
            return []

        # Structure: results.DisorderGeneAssociation[]
        gene_assoc_list = results.get("DisorderGeneAssociation", [])
        if not isinstance(gene_assoc_list, list):
            return []

        genes = []
        for item in gene_assoc_list:
            gene = item.get("Gene", {})
            symbol = gene.get("Symbol", "")
            name = gene.get("Name", "")
            gtype = item.get("DisorderGeneAssociationType", "")

            # Get locus from Gene.Locus list
            locus_list = gene.get("Locus", [])
            locus = None
            if isinstance(locus_list, list) and locus_list:
                locus = locus_list[0].get("GeneLocus", None) if isinstance(locus_list[0], dict) else None

            if symbol:
                genes.append(GeneAssociation(
                    gene_symbol=symbol,
                    gene_name=name,
                    gene_type=gtype or None,
                    locus=locus,
                ))
        return genes

    def get_natural_history(self, orpha_code: int) -> dict:
        """Get age of onset, inheritance for a disease."""
        results = self._get_results(f"/rd-natural_history/orphacodes/{orpha_code}")
        if not isinstance(results, dict):
            return {}
        return results

    def get_epidemiology(self, orpha_code: int) -> dict:
        """Get prevalence data for a disease."""
        results = self._get_results(f"/rd-epidemiology/orphacodes/{orpha_code}")
        if not isinstance(results, dict):
            return {}
        return results

    def close(self):
        self.client.close()


def fetch_disease(client: OrphanetClient, orpha_code: int, name: str) -> RareDisease:
    """Fetch all available data for a single disease."""
    disease = RareDisease(orpha_code=orpha_code, name=name)

    # Cross-references (OMIM, ICD-10)
    xref = client.get_cross_references(orpha_code)
    if xref and isinstance(xref, dict):
        refs = xref.get("ExternalReference") or []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            source = ref.get("Source", "")
            reference = ref.get("Reference", "")
            if source == "OMIM" and reference:
                disease.omim_ids.append(reference)
            elif source == "ICD-10" and reference:
                disease.icd10_codes.append(reference)

    # Phenotypes
    disease.hpo_associations = client.get_phenotypes(orpha_code)

    # Genes
    disease.genes = client.get_genes(orpha_code)

    # Natural history
    nat_hist = client.get_natural_history(orpha_code)
    if nat_hist:
        inheritance = nat_hist.get("TypeOfInheritance", [])
        if isinstance(inheritance, list):
            disease.inheritance = [i for i in inheritance if isinstance(i, str)]
        age_onset = nat_hist.get("AverageAgeOfOnset", [])
        if isinstance(age_onset, list):
            disease.age_of_onset = [a for a in age_onset if isinstance(a, str)]

    return disease


def fetch_diseases(
    max_diseases: int = 100,
    output_path: Optional[Path] = None,
) -> list[RareDisease]:
    """Fetch disease records from Orphadata API.

    Args:
        max_diseases: Maximum number of diseases to fetch (0 = all).
        output_path: Where to save the JSON output.
    """
    output_path = output_path or RAW_DIR / "diseases.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = OrphanetClient()
    try:
        print("Fetching ORPHAcode index...")
        all_codes = client.get_all_orphacodes()
        print(f"  Found {len(all_codes)} ORPHAcodes in index")

        if not all_codes:
            print("ERROR: No ORPHAcodes returned. Check API availability.")
            return []

        # Limit if requested
        if max_diseases > 0:
            all_codes = all_codes[:max_diseases]

        diseases: list[RareDisease] = []
        for i, entry in enumerate(all_codes):
            if not isinstance(entry, dict):
                continue
            code = entry.get("ORPHAcode", 0)
            name = entry.get("Preferred term", f"ORPHA:{code}")
            code = int(code) if code else 0
            if code == 0:
                continue

            print(f"  [{i+1}/{len(all_codes)}] Fetching ORPHA:{code} -- {name}")
            disease = fetch_disease(client, code, str(name))
            diseases.append(disease)

            # Periodic save
            if (i + 1) % 50 == 0:
                _save_diseases(diseases, output_path)
                print(f"  Saved {len(diseases)} diseases so far")

        _save_diseases(diseases, output_path)
        print(f"\nDone. Saved {len(diseases)} diseases to {output_path}")
        return diseases

    finally:
        client.close()


def _save_diseases(diseases: list[RareDisease], path: Path):
    with open(path, "w") as f:
        json.dump([d.model_dump() for d in diseases], f, indent=2)


def load_diseases(path: Optional[Path] = None) -> list[RareDisease]:
    """Load previously fetched disease records from JSON."""
    path = path or RAW_DIR / "diseases.json"
    if not path.exists():
        raise FileNotFoundError(f"No disease data at {path}. Run fetch_diseases() first.")
    with open(path) as f:
        data = json.load(f)
    return [RareDisease(**d) for d in data]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Orphanet rare disease data")
    parser.add_argument("-n", "--max-diseases", type=int, default=100,
                        help="Max diseases to fetch (0=all)")
    args = parser.parse_args()
    fetch_diseases(max_diseases=args.max_diseases)
