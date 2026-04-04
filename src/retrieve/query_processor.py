"""Query processing: extract phenotypes, map to HPO, generate multi-queries.

Transforms a clinical vignette into optimised retrieval queries.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class ProcessedQuery(BaseModel):
    """Output of query processing."""
    original_vignette: str
    extracted_phenotypes: list[str] = []
    hpo_candidates: list[dict] = []  # [{"hpo_id": "HP:...", "name": "...", "score": 0.9}]
    retrieval_queries: list[str] = []
    hyde_document: str = ""  # Hypothetical document for HyDE retrieval


def extract_phenotypes_with_llm(vignette: str, client=None) -> list[str]:
    """Use Claude to extract key phenotypic features from a clinical vignette."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1000,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                "Extract the key phenotypic features from this clinical vignette. "
                "Return a JSON array of short phenotype descriptions.\n\n"
                f"Vignette: {vignette}\n\n"
                "Return ONLY a JSON array like: "
                '[\"progressive cerebellar ataxia\", \"elevated AFP\", ...]'
            ),
        }],
    )
    text = response.content[0].text.strip()
    # Parse JSON from response
    if text.startswith("["):
        return json.loads(text)
    # Try to find JSON array in response
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        return json.loads(text[start:end])
    return []


def generate_retrieval_queries(vignette: str, phenotypes: list[str],
                                client=None) -> list[str]:
    """Generate diverse retrieval queries from a vignette and extracted phenotypes."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    prompt_template = (PROMPT_DIR / "query_expansion.md").read_text()
    prompt = prompt_template.replace("{vignette}", vignette)

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        temperature=0.2,
        messages=[{
            "role": "user",
            "content": prompt,
        }],
    )
    text = response.content[0].text.strip()

    # Parse queries from JSON response
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            queries = [q["query"] for q in data.get("queries", [])]
            if queries:
                return queries
    except (json.JSONDecodeError, KeyError):
        pass

    # Fallback: generate basic queries from phenotypes
    return _fallback_queries(vignette, phenotypes)


def generate_hyde_document(vignette: str, client=None) -> str:
    """Generate a hypothetical Orphanet-style disease entry from a clinical vignette.

    HyDE (Hypothetical Document Embedding) bridges the semantic gap between
    clinical presentations and structured knowledge base entries by generating
    a synthetic document that resembles what the correct KB entry would look like.
    """
    if client is None:
        import anthropic
        client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        temperature=0.0,
        messages=[{
            "role": "user",
            "content": (
                "Based on this clinical vignette, write a hypothetical rare disease "
                "knowledge base entry that would match this patient's presentation. "
                "Use this format:\n\n"
                "Disease: [most likely disease name]\n"
                "Synonyms: [alternative names]\n"
                "Inheritance: [pattern]\n"
                "Age of onset: [typical onset]\n"
                "Clinical features:\n"
                "- [feature 1]\n"
                "- [feature 2]\n"
                "Associated genes: [if applicable]\n\n"
                "Write ONLY the disease entry, no explanation.\n\n"
                f"Vignette: {vignette}"
            ),
        }],
    )
    return response.content[0].text.strip()


def _fallback_queries(vignette: str, phenotypes: list[str]) -> list[str]:
    """Generate basic retrieval queries without LLM."""
    queries = []

    # Query 1: All phenotypes
    if phenotypes:
        queries.append(" ".join(phenotypes))

    # Query 2: Original vignette (truncated)
    queries.append(vignette[:500])

    # Query 3: Pairs of most distinctive features
    if len(phenotypes) >= 2:
        queries.append(f"{phenotypes[0]} AND {phenotypes[1]} rare disease")

    # Query 4: Individual features
    for p in phenotypes[:3]:
        queries.append(f"{p} rare genetic disease differential diagnosis")

    return queries


def process_query(vignette: str, hpo_ontology=None, use_llm: bool = True,
                  client=None, use_hyde: bool = False) -> ProcessedQuery:
    """Full query processing pipeline."""
    result = ProcessedQuery(original_vignette=vignette)

    # Step 1: Extract phenotypes
    if use_llm:
        result.extracted_phenotypes = extract_phenotypes_with_llm(vignette, client)
    else:
        # Simple keyword extraction fallback
        result.extracted_phenotypes = [vignette]

    # Step 2: Map to HPO (if ontology available)
    if hpo_ontology is not None:
        for phenotype in result.extracted_phenotypes:
            matches = hpo_ontology.search_by_name(phenotype, max_results=1)
            if matches:
                hpo_id, name, score = matches[0]
                result.hpo_candidates.append({
                    "hpo_id": hpo_id,
                    "name": name,
                    "score": score,
                    "original": phenotype,
                })

    # Step 3: Generate retrieval queries
    if use_llm:
        result.retrieval_queries = generate_retrieval_queries(
            vignette, result.extracted_phenotypes, client
        )
    else:
        result.retrieval_queries = _fallback_queries(vignette, result.extracted_phenotypes)

    # Step 4: Generate HyDE document (hypothetical KB entry)
    if use_hyde and use_llm:
        try:
            result.hyde_document = generate_hyde_document(vignette, client)
        except Exception as e:
            print(f"  HyDE generation failed: {e}")

    return result
