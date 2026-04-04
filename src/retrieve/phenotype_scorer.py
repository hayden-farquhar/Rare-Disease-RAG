"""HPO phenotype overlap scoring for disease candidate ranking.

Uses HPO ontology hierarchy to compute semantic phenotype overlap
between query phenotypes and disease phenotype profiles.
Acts as a rescue mechanism when embedding-based retrieval misses.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.ingest.hpo_mapper import HPOOntology
from src.ingest.orphanet_ingest import RareDisease

COUNTS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "hpo_disease_counts.json"


class PhenotypeScorer:
    """Score diseases by HPO phenotype overlap with query phenotypes."""

    def __init__(self, hpo: HPOOntology, diseases: list[RareDisease]):
        self.hpo = hpo
        self.disease_hpo: dict[int, set[str]] = {}
        self.hpo_disease_count: dict[str, int] = defaultdict(int)
        self.n_diseases = len(diseases)

        # Build disease -> HPO term sets
        for d in diseases:
            terms = {h.hpo_id for h in d.hpo_associations}
            self.disease_hpo[d.orpha_code] = terms
            for t in terms:
                self.hpo_disease_count[t] += 1

        # Build disease name lookup
        self.disease_names: dict[int, str] = {d.orpha_code: d.name for d in diseases}

    def information_content(self, hpo_id: str) -> float:
        """Compute information content of an HPO term.

        IC = -log(P(term)) where P = fraction of diseases with this term.
        Rarer terms have higher IC (more informative).
        """
        count = self.hpo_disease_count.get(hpo_id, 0)
        if count == 0:
            return 10.0  # Very rare / unknown term gets high IC
        p = count / self.n_diseases
        return -math.log2(p)

    def score_disease(
        self, query_hpo_ids: list[str], disease_orpha_code: int,
        use_ancestors: bool = True, ancestor_depth: int = 2,
    ) -> float:
        """Score a single disease against query phenotypes.

        Returns a score between 0 and 1 based on IC-weighted overlap.
        """
        disease_terms = self.disease_hpo.get(disease_orpha_code, set())
        if not disease_terms or not query_hpo_ids:
            return 0.0

        total_ic = 0.0
        matched_ic = 0.0

        for q_hpo in query_hpo_ids:
            ic = self.information_content(q_hpo)
            total_ic += ic

            # Direct match
            if q_hpo in disease_terms:
                matched_ic += ic
                continue

            # Ancestor match (partial credit)
            if use_ancestors:
                q_ancestors = set(self.hpo.get_ancestors(q_hpo, max_depth=ancestor_depth))
                d_ancestors = set()
                for d_term in disease_terms:
                    d_ancestors.update(self.hpo.get_ancestors(d_term, max_depth=ancestor_depth))

                # Check if query term is an ancestor of a disease term
                if q_hpo in d_ancestors:
                    matched_ic += ic * 0.7  # Partial credit
                    continue

                # Check shared ancestors
                shared = q_ancestors & d_ancestors
                if shared:
                    # Credit proportional to specificity of shared ancestor
                    best_shared_ic = max(self.information_content(a) for a in shared)
                    matched_ic += ic * min(0.5, best_shared_ic / ic) if ic > 0 else 0
                    continue

        return matched_ic / total_ic if total_ic > 0 else 0.0

    def rank_diseases(
        self, query_hpo_ids: list[str], top_k: int = 20
    ) -> list[tuple[int, str, float]]:
        """Rank all diseases by phenotype overlap score.

        Returns list of (orpha_code, disease_name, score) tuples.
        """
        scored = []
        for orpha_code, terms in self.disease_hpo.items():
            score = self.score_disease(query_hpo_ids, orpha_code)
            if score > 0:
                scored.append((orpha_code, self.disease_names[orpha_code], score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    def save_counts(self, path: Optional[Path] = None):
        """Save HPO-disease count index."""
        path = path or COUNTS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(self.hpo_disease_count), f)

    def get_rescue_candidates(
        self, query_hpo_ids: list[str],
        already_retrieved: set[str],
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> list[tuple[int, str, float]]:
        """Get disease candidates that the retriever missed but have high phenotype overlap.

        This is the rescue mechanism — inject these into the retrieval results
        if they score above min_score but weren't in the top-k dense results.
        """
        ranked = self.rank_diseases(query_hpo_ids, top_k=top_k + len(already_retrieved))
        rescue = []
        for orpha_code, name, score in ranked:
            if score < min_score:
                break
            if name not in already_retrieved:
                rescue.append((orpha_code, name, score))
                if len(rescue) >= top_k:
                    break
        return rescue
