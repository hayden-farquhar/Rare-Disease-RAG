"""Generate narrative clinical descriptions from structured HPO phenotype data.

Uses Claude Haiku to transform HPO term lists into natural clinical prose
that embeds well alongside clinical vignettes.
"""

import json
import time
from pathlib import Path
from typing import Optional

from src.ingest.orphanet_ingest import RareDisease

NARRATIVES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "narratives.json"

NARRATIVE_PROMPT = """You are a medical textbook author. Given structured phenotype data for a rare disease, write a concise clinical description (100-150 words) of how a typical patient presents. Write in clinical prose, as if describing the disease in a medical reference.

Include:
- Age of onset and inheritance pattern
- The most characteristic presenting features
- Key distinguishing clinical findings
- Disease progression if relevant
- Associated complications

Do NOT use HPO codes. Write in natural clinical language.

Disease: {name}
Inheritance: {inheritance}
Age of onset: {onset}
Associated genes: {genes}

Very frequent features (present in >80% of patients):
{vf_features}

Frequent features (present in 30-80% of patients):
{freq_features}

Write ONLY the clinical description paragraph, no title or headers."""


def build_disease_prompt(disease: RareDisease) -> str:
    """Build a narrative generation prompt for a single disease."""
    vf = [h for h in disease.hpo_associations if h.frequency and "99-80" in h.frequency]
    freq = [h for h in disease.hpo_associations if h.frequency and "79-30" in h.frequency]

    vf_text = "\n".join(f"- {h.hpo_term}" for h in vf[:20]) or "- None documented"
    freq_text = "\n".join(f"- {h.hpo_term}" for h in freq[:15]) or "- None documented"

    return NARRATIVE_PROMPT.format(
        name=disease.name,
        inheritance=", ".join(disease.inheritance) or "Unknown",
        onset=", ".join(disease.age_of_onset) or "Unknown",
        genes=", ".join(g.gene_symbol for g in disease.genes[:5]) or "Unknown",
        vf_features=vf_text,
        freq_features=freq_text,
    )


def generate_narratives_batch(
    diseases: list[RareDisease],
    client,
    model: str = "claude-haiku-4-5-20251001",
    batch_size: int = 1,
    delay: float = 0.2,
    max_retries: int = 3,
) -> dict[int, str]:
    """Generate narrative descriptions for a batch of diseases.

    Returns dict mapping orpha_code to narrative text.
    """
    results = {}

    for i, disease in enumerate(diseases):
        prompt = build_disease_prompt(disease)

        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=300,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                narrative = response.content[0].text.strip()
                results[disease.orpha_code] = narrative
                break
            except Exception as e:
                if "overloaded" in str(e).lower() or "529" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"    API overloaded, retrying in {wait}s...")
                    time.sleep(wait)
                elif "rate" in str(e).lower():
                    time.sleep(5)
                else:
                    print(f"    Error for ORPHA:{disease.orpha_code}: {e}")
                    break

        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{len(diseases)} narratives")

        time.sleep(delay)

    return results


def load_narratives(path: Optional[Path] = None) -> dict[int, str]:
    """Load previously generated narratives."""
    path = path or NARRATIVES_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    # Keys are strings in JSON, convert to int
    return {int(k): v for k, v in data.items()}


def save_narratives(narratives: dict[int, str], path: Optional[Path] = None):
    """Save narratives to JSON."""
    path = path or NARRATIVES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in narratives.items()}, f, indent=2)
