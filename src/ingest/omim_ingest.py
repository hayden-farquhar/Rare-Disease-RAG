"""OMIM data ingestion (stub).

OMIM requires registration for data download (https://omim.org/downloads).
This module handles parsing OMIM text dump files once downloaded.

For Phase 1, we focus on Orphanet data. OMIM will be added in Phase 2
once the data access agreement is in place.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "omim"


class OMIMEntry(BaseModel):
    mim_number: str
    title: str
    clinical_synopsis: dict = {}  # System -> list of features
    text_description: str = ""
    molecular_genetics: str = ""
    differential_diagnosis: str = ""
    gene_symbols: list[str] = []
    inheritance: list[str] = []


def parse_omim_dump(path: Path) -> list[OMIMEntry]:
    """Parse OMIM genemap2.txt or morbidmap.txt file.

    TODO: Implement once OMIM data access is obtained.
    """
    raise NotImplementedError(
        "OMIM parsing not yet implemented. "
        "Register at https://omim.org/downloads to obtain data files."
    )


def load_omim_entries(path: Optional[Path] = None) -> list[OMIMEntry]:
    """Load previously parsed OMIM entries."""
    path = path or RAW_DIR / "omim_entries.json"
    if not path.exists():
        raise FileNotFoundError(f"No OMIM data at {path}")
    with open(path) as f:
        return [OMIMEntry(**e) for e in json.load(f)]
