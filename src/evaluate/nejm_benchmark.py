"""NEJM Clinical Problem-Solving benchmark management.

Handles loading, validation, and evaluation of NEJM diagnostic challenge cases.
Cases must be manually curated (see data/benchmarks/README.md).
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"


class NEJMCase(BaseModel):
    """Structured representation of an NEJM diagnostic challenge case."""
    case_id: str
    pmid: str = ""
    year: Optional[int] = None
    title: str = ""
    clinical_vignette: str  # Text up to diagnostic reveal
    final_diagnosis: str
    orpha_code: Optional[str] = None
    omim_id: Optional[str] = None
    rarity: str = "rare"  # "ultra-rare", "rare", "uncommon"
    systems_involved: list[str] = []
    key_discriminating_features: list[str] = []
    difficulty_rating: str = "moderate"  # "easy", "moderate", "hard"
    aliases: list[str] = []  # Alternative names for the diagnosis


def load_benchmark(path: Optional[Path] = None) -> list[NEJMCase]:
    """Load benchmark cases from JSON."""
    path = path or BENCHMARK_DIR / "nejm_cases.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No benchmark file at {path}. "
            "NEJM cases must be manually curated."
        )
    with open(path) as f:
        data = json.load(f)
    return [NEJMCase(**c) for c in data]


def save_benchmark(cases: list[NEJMCase], path: Optional[Path] = None):
    """Save benchmark cases to JSON."""
    path = path or BENCHMARK_DIR / "nejm_cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([c.model_dump() for c in cases], f, indent=2)
    print(f"Saved {len(cases)} benchmark cases to {path}")
