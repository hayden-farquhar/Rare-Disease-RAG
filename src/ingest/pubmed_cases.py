"""Fetch rare disease case reports from PubMed/PMC.

Uses the NCBI E-utilities API to search and retrieve case reports.
Focuses on the PMC Open Access subset for full-text access.
"""

import json
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "pubmed_cases"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class CaseReport(BaseModel):
    pmid: str
    title: str
    abstract: str = ""
    journal: str = ""
    year: Optional[int] = None
    mesh_terms: list[str] = []
    keywords: list[str] = []
    full_text: Optional[str] = None  # Only for PMC OA articles


def search_pubmed(
    query: str,
    max_results: int = 1000,
    api_key: Optional[str] = None,
) -> list[str]:
    """Search PubMed and return PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": min(max_results, 10000),
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    resp = httpx.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    pmids = data.get("esearchresult", {}).get("idlist", [])
    total = int(data.get("esearchresult", {}).get("count", 0))
    print(f"PubMed search returned {total} results, fetching {len(pmids)}")
    return pmids


def fetch_article_metadata(
    pmids: list[str],
    batch_size: int = 100,
    api_key: Optional[str] = None,
) -> list[CaseReport]:
    """Fetch article metadata for a list of PMIDs."""
    cases = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if api_key:
            params["api_key"] = api_key

        resp = httpx.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=30.0)
        resp.raise_for_status()

        # Parse XML response
        from lxml import etree
        root = etree.fromstring(resp.content)

        for article in root.findall(".//PubmedArticle"):
            try:
                pmid_el = article.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else ""

                title_el = article.find(".//ArticleTitle")
                title = "".join(title_el.itertext()) if title_el is not None else ""

                abstract_parts = []
                for abs_text in article.findall(".//AbstractText"):
                    label = abs_text.get("Label", "")
                    text = "".join(abs_text.itertext())
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = "\n".join(abstract_parts)

                journal_el = article.find(".//Journal/Title")
                journal = journal_el.text if journal_el is not None else ""

                year_el = article.find(".//PubDate/Year")
                year = int(year_el.text) if year_el is not None else None

                mesh_terms = []
                for mesh in article.findall(".//MeshHeading/DescriptorName"):
                    mesh_terms.append(mesh.text or "")

                keywords = []
                for kw in article.findall(".//Keyword"):
                    keywords.append(kw.text or "")

                cases.append(CaseReport(
                    pmid=pmid, title=title, abstract=abstract,
                    journal=journal, year=year,
                    mesh_terms=mesh_terms, keywords=keywords,
                ))
            except Exception as e:
                print(f"  Error parsing article: {e}")
                continue

        print(f"  Fetched metadata for {min(i + batch_size, len(pmids))}/{len(pmids)} articles")
        time.sleep(0.34)  # NCBI rate limit: 3 requests/sec without API key

    return cases


def fetch_rare_disease_cases(
    max_cases: int = 500,
    output_path: Optional[Path] = None,
    api_key: Optional[str] = None,
) -> list[CaseReport]:
    """Fetch rare disease case reports from PubMed."""
    output_path = output_path or RAW_DIR / "case_reports.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = (
        '("case reports"[pt] OR "case report"[ti]) '
        'AND ("rare diseases"[MeSH] OR "genetic diseases, inborn"[MeSH] '
        'OR "metabolism, inborn errors"[MeSH]) '
        'AND ("2015"[pdat] : "2026"[pdat]) '
        'AND english[la]'
    )

    print("Searching PubMed for rare disease case reports...")
    pmids = search_pubmed(query, max_results=max_cases, api_key=api_key)

    if not pmids:
        print("No PMIDs returned.")
        return []

    print(f"Fetching metadata for {len(pmids)} articles...")
    cases = fetch_article_metadata(pmids, api_key=api_key)

    # Save
    with open(output_path, "w") as f:
        json.dump([c.model_dump() for c in cases], f, indent=2)
    print(f"Saved {len(cases)} case reports to {output_path}")

    return cases


def load_cases(path: Optional[Path] = None) -> list[CaseReport]:
    """Load previously fetched case reports."""
    path = path or RAW_DIR / "case_reports.json"
    if not path.exists():
        raise FileNotFoundError(f"No case data at {path}. Run fetch_rare_disease_cases() first.")
    with open(path) as f:
        data = json.load(f)
    return [CaseReport(**d) for d in data]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch PubMed rare disease case reports")
    parser.add_argument("-n", "--max-cases", type=int, default=500)
    args = parser.parse_args()
    fetch_rare_disease_cases(max_cases=args.max_cases)
