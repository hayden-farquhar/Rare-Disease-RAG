"""Curate 30 additional ultra-rare benchmark cases from PMC.

Workflow:
1. Get Orphanet diseases NOT already in the benchmark
2. Filter to those with good HPO coverage (>5 HPO terms)
3. Search PMC for case reports for each disease
4. Use Claude to generate clinical vignettes from abstracts
5. Save to ultra_rare_cases_new.json
"""

import json
import sys
import time
import random
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from src.ingest.orphanet_ingest import load_diseases

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def get_existing_orpha_codes():
    """Get orpha codes already in benchmark."""
    codes = set()
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json"]:
        fpath = BENCHMARK_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                for case in json.load(f):
                    if "orpha_code" in case:
                        codes.add(str(case["orpha_code"]))
    return codes


def search_pmc_case_report(disease_name: str) -> list[dict]:
    """Search PMC for case reports about a disease. Returns list of {pmid, title, abstract}."""
    query = f'("{disease_name}"[Title/Abstract]) AND ("case report"[Publication Type] OR "case reports"[Publication Type]) AND english[la]'

    try:
        # Search
        resp = httpx.get(f"{EUTILS_BASE}/esearch.fcgi", params={
            "db": "pubmed", "term": query, "retmax": 5, "retmode": "json",
        }, timeout=15.0)
        resp.raise_for_status()
        pmids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            return []

        time.sleep(0.4)

        # Fetch abstracts
        resp = httpx.get(f"{EUTILS_BASE}/efetch.fcgi", params={
            "db": "pubmed", "id": ",".join(pmids), "retmode": "xml",
        }, timeout=30.0)
        resp.raise_for_status()

        from lxml import etree
        root = etree.fromstring(resp.content)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.findtext(".//PMID", "")
            title = article.findtext(".//ArticleTitle", "")
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(t.text or "" for t in abstract_parts)
            if abstract and len(abstract) > 200:
                articles.append({"pmid": pmid, "title": title, "abstract": abstract})
        return articles

    except Exception as e:
        print(f"  PMC search failed for {disease_name}: {e}")
        return []


def generate_vignette(disease_name: str, abstract: str, client) -> dict:
    """Use Claude to generate a clinical vignette from a case report abstract."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                "Convert this case report abstract into a clinical vignette suitable for "
                "diagnostic reasoning. The vignette should:\n"
                "- Present the patient's demographics, history, and findings\n"
                "- Include specific clinical details (measurements, lab values, imaging)\n"
                "- NOT mention the diagnosis or disease name anywhere\n"
                "- Read like a clinical problem-solving challenge\n"
                "- Be 150-250 words\n\n"
                f"Disease (DO NOT include in vignette): {disease_name}\n"
                f"Abstract: {abstract}\n\n"
                "Return ONLY the clinical vignette text, nothing else."
            ),
        }],
    )
    vignette = response.content[0].text.strip()

    # Verify disease name isn't leaked
    if disease_name.lower() in vignette.lower():
        # Try once more with stronger instruction
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    f"Rewrite this clinical vignette to remove ALL mentions of "
                    f"'{disease_name}' and any related disease names. Replace with "
                    f"descriptive clinical terms only.\n\n{vignette}"
                ),
            }],
        )
        vignette = response.content[0].text.strip()

    return vignette


def extract_features(vignette: str, client) -> list[str]:
    """Extract key discriminating features from a vignette."""
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                "Extract 5-8 key discriminating clinical features from this vignette. "
                "Return as a JSON array of short clinical terms.\n\n"
                f"Vignette: {vignette}\n\n"
                'Return ONLY a JSON array like: ["feature1", "feature2", ...]'
            ),
        }],
    )
    text = response.content[0].text.strip()
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return []


def main():
    print("=" * 70)
    print("Curating Additional Ultra-Rare Benchmark Cases")
    print("=" * 70)

    # Load all diseases from KB
    all_diseases = load_diseases()
    print(f"Loaded {len(all_diseases)} diseases from Orphanet")

    # Filter to candidates
    existing_codes = get_existing_orpha_codes()
    print(f"Existing benchmark diseases: {len(existing_codes)}")

    candidates = [
        d for d in all_diseases
        if str(d.orpha_code) not in existing_codes
        and len(d.hpo_associations) >= 5  # enough phenotypes for meaningful retrieval
    ]
    print(f"Candidate diseases (not in benchmark, >=5 HPO terms, has description): {len(candidates)}")

    # Shuffle and take more than needed (some will fail PMC search)
    random.seed(42)
    random.shuffle(candidates)

    import anthropic
    client = anthropic.Anthropic()

    new_cases = []
    attempted = 0
    target = 30

    for disease in candidates:
        if len(new_cases) >= target:
            break
        attempted += 1

        disease_name = disease.name
        orpha_code = disease.orpha_code
        print(f"\n[{len(new_cases)+1}/{target}] Trying: {disease_name} (ORPHA:{orpha_code})")

        # Search PMC
        articles = search_pmc_case_report(disease_name)
        if not articles:
            print(f"  No case reports found, skipping")
            continue

        # Use best article (longest abstract)
        best = max(articles, key=lambda a: len(a["abstract"]))
        print(f"  Found case report: PMID {best['pmid']} ({len(best['abstract'])} chars)")

        # Generate vignette
        try:
            vignette = generate_vignette(disease_name, best["abstract"], client)
        except Exception as e:
            print(f"  Vignette generation failed: {e}")
            continue

        if len(vignette) < 100:
            print(f"  Vignette too short ({len(vignette)} chars), skipping")
            continue

        # Check for diagnosis leakage
        if disease_name.lower() in vignette.lower():
            print(f"  Diagnosis leaked in vignette, skipping")
            continue

        # Extract features
        try:
            features = extract_features(vignette, client)
        except Exception as e:
            features = []

        # Build case
        case = {
            "case_id": f"curated_{orpha_code}",
            "pmc_id": best["pmid"],
            "title": f"Curated: {disease_name}",
            "clinical_vignette": vignette,
            "final_diagnosis": disease_name,
            "orpha_code": str(orpha_code),
            "rarity": "ultra-rare",
            "key_discriminating_features": features,
            "difficulty_rating": "hard",
            "aliases": list(disease.synonyms) if disease.synonyms else [],
            "source": "pmc_curated",
        }
        new_cases.append(case)
        print(f"  SUCCESS: {len(vignette)} chars, {len(features)} features")

        # Save incrementally
        out_path = BENCHMARK_DIR / "ultra_rare_cases_new.json"
        with open(out_path, "w") as f:
            json.dump(new_cases, f, indent=2)

        time.sleep(0.5)

    print(f"\n{'='*70}")
    print(f"CURATED {len(new_cases)} new cases (attempted {attempted} diseases)")
    print(f"{'='*70}")

    # Save final
    out_path = BENCHMARK_DIR / "ultra_rare_cases_new.json"
    with open(out_path, "w") as f:
        json.dump(new_cases, f, indent=2)
    print(f"Saved to {out_path}")

    # Print summary
    for case in new_cases:
        print(f"  {case['case_id']}: {case['final_diagnosis']} ({len(case['clinical_vignette'])} chars)")


if __name__ == "__main__":
    main()
