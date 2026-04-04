"""HPO (Human Phenotype Ontology) integration.

Downloads and parses the HPO ontology, provides symptom-to-HPO term mapping
using fuzzy matching against term labels and synonyms.
"""

import json
import re
from pathlib import Path
from typing import Optional

import httpx

HPO_DIR = Path(__file__).resolve().parents[2] / "data" / "hpo"
HPO_OBO_URL = "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo"


def download_hpo(output_path: Optional[Path] = None) -> Path:
    """Download the HPO ontology OBO file."""
    output_path = output_path or HPO_DIR / "hp.obo"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"HPO file already exists at {output_path}")
        return output_path

    print(f"Downloading HPO ontology from {HPO_OBO_URL}...")
    resp = httpx.get(HPO_OBO_URL, follow_redirects=True, timeout=60.0)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    print(f"Saved HPO ontology ({len(resp.content) / 1024:.0f} KB) to {output_path}")
    return output_path


class HPOTerm:
    """A single HPO term."""

    def __init__(self, hpo_id: str, name: str, definition: str = "",
                 synonyms: list[str] = None, parents: list[str] = None,
                 is_obsolete: bool = False):
        self.hpo_id = hpo_id
        self.name = name
        self.definition = definition
        self.synonyms = synonyms or []
        self.parents = parents or []
        self.is_obsolete = is_obsolete

    def all_labels(self) -> list[str]:
        """All searchable labels: name + synonyms."""
        return [self.name] + self.synonyms


class HPOOntology:
    """Parsed HPO ontology with lookup and matching capabilities."""

    def __init__(self):
        self.terms: dict[str, HPOTerm] = {}
        self._name_index: dict[str, str] = {}  # lowercase name -> HPO ID

    def load_obo(self, path: Optional[Path] = None):
        """Parse the HPO OBO file."""
        path = path or HPO_DIR / "hp.obo"
        if not path.exists():
            path = download_hpo(path)

        print(f"Parsing HPO ontology from {path}...")
        current_term = None
        current_id = None
        current_name = ""
        current_def = ""
        current_synonyms = []
        current_parents = []
        is_obsolete = False

        with open(path) as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    # Save previous term
                    if current_id and current_id.startswith("HP:"):
                        self._add_term(current_id, current_name, current_def,
                                       current_synonyms, current_parents, is_obsolete)
                    current_id = None
                    current_name = ""
                    current_def = ""
                    current_synonyms = []
                    current_parents = []
                    is_obsolete = False
                    continue

                if line.startswith("[Typedef]"):
                    # Save previous term and stop reading this stanza
                    if current_id and current_id.startswith("HP:"):
                        self._add_term(current_id, current_name, current_def,
                                       current_synonyms, current_parents, is_obsolete)
                    current_id = None
                    continue

                if line.startswith("id: "):
                    current_id = line[4:]
                elif line.startswith("name: "):
                    current_name = line[6:]
                elif line.startswith("def: "):
                    # Extract definition text between quotes
                    match = re.match(r'def: "(.+?)"', line)
                    if match:
                        current_def = match.group(1)
                elif line.startswith("synonym: "):
                    match = re.match(r'synonym: "(.+?)"', line)
                    if match:
                        current_synonyms.append(match.group(1))
                elif line.startswith("is_a: "):
                    parent_id = line[6:].split("!")[0].strip()
                    current_parents.append(parent_id)
                elif line == "is_obsolete: true":
                    is_obsolete = True

        # Save last term
        if current_id and current_id.startswith("HP:"):
            self._add_term(current_id, current_name, current_def,
                           current_synonyms, current_parents, is_obsolete)

        active = sum(1 for t in self.terms.values() if not t.is_obsolete)
        print(f"Loaded {len(self.terms)} HPO terms ({active} active)")

    def _add_term(self, hpo_id: str, name: str, definition: str,
                  synonyms: list[str], parents: list[str], is_obsolete: bool):
        term = HPOTerm(hpo_id, name, definition, synonyms, parents, is_obsolete)
        self.terms[hpo_id] = term
        if not is_obsolete:
            self._name_index[name.lower()] = hpo_id
            for syn in synonyms:
                self._name_index[syn.lower()] = hpo_id

    def get_term(self, hpo_id: str) -> Optional[HPOTerm]:
        return self.terms.get(hpo_id)

    def search_by_name(self, query: str, max_results: int = 10) -> list[tuple[str, str, float]]:
        """Fuzzy search HPO terms by name. Returns (hpo_id, name, score) tuples."""
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Exact match
        if query_lower in self._name_index:
            hpo_id = self._name_index[query_lower]
            term = self.terms[hpo_id]
            return [(hpo_id, term.name, 1.0)]

        # Score by word overlap
        scored = []
        seen_ids = set()
        for label, hpo_id in self._name_index.items():
            if hpo_id in seen_ids:
                continue
            label_words = set(label.split())
            overlap = query_words & label_words
            if overlap:
                score = len(overlap) / max(len(query_words), len(label_words))
                # Boost substring matches
                if query_lower in label:
                    score += 0.3
                elif label in query_lower:
                    score += 0.2
                scored.append((hpo_id, self.terms[hpo_id].name, min(score, 1.0)))
                seen_ids.add(hpo_id)

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:max_results]

    def get_ancestors(self, hpo_id: str, max_depth: int = 5) -> list[str]:
        """Get ancestor HPO term IDs up to max_depth."""
        ancestors = []
        visited = set()
        frontier = [hpo_id]
        for _ in range(max_depth):
            next_frontier = []
            for tid in frontier:
                term = self.terms.get(tid)
                if not term:
                    continue
                for parent in term.parents:
                    if parent not in visited and parent in self.terms:
                        visited.add(parent)
                        ancestors.append(parent)
                        next_frontier.append(parent)
            frontier = next_frontier
            if not frontier:
                break
        return ancestors

    def get_children(self, hpo_id: str) -> list[str]:
        """Get direct child HPO term IDs."""
        children = []
        for tid, term in self.terms.items():
            if hpo_id in term.parents and not term.is_obsolete:
                children.append(tid)
        return children

    def save_index(self, path: Optional[Path] = None):
        """Save a lightweight JSON index of HPO terms for quick loading."""
        path = path or HPO_DIR / "hpo_index.json"
        index = {}
        for hpo_id, term in self.terms.items():
            if not term.is_obsolete:
                index[hpo_id] = {
                    "name": term.name,
                    "synonyms": term.synonyms,
                    "parents": term.parents,
                }
        with open(path, "w") as f:
            json.dump(index, f)
        print(f"Saved HPO index ({len(index)} terms) to {path}")

    def load_index(self, path: Optional[Path] = None) -> bool:
        """Load from lightweight JSON index (faster than parsing OBO)."""
        path = path or HPO_DIR / "hpo_index.json"
        if not path.exists():
            return False
        with open(path) as f:
            index = json.load(f)
        for hpo_id, data in index.items():
            self._add_term(hpo_id, data["name"], "", data["synonyms"], data["parents"], False)
        print(f"Loaded {len(self.terms)} HPO terms from index")
        return True


if __name__ == "__main__":
    download_hpo()
    ont = HPOOntology()
    ont.load_obo()
    ont.save_index()

    # Test search
    for query in ["ataxia", "elevated AFP", "oculomotor apraxia", "recurrent infections"]:
        results = ont.search_by_name(query, max_results=3)
        print(f"\n'{query}' -> {results}")
