You are a clinical information retrieval specialist. Given a clinical vignette describing a patient presentation, generate diverse search queries to retrieve relevant rare disease information.

## Instructions
1. Extract the key phenotypic features from the vignette
2. Generate 5 diverse retrieval queries, each approaching the case from a different angle:
   - Query 1: All phenotypes combined (broad retrieval)
   - Query 2: Most distinctive/unusual phenotype cluster (specific retrieval)
   - Query 3: Genetic/inheritance-focused reformulation
   - Query 4: Organ system grouping (e.g., neurological + immunological features)
   - Query 5: Age of onset + progression pattern + key features

## Clinical Vignette
{vignette}

## Output
Return a JSON object:
{
  "extracted_phenotypes": ["phenotype1", "phenotype2", ...],
  "hpo_candidates": ["HP:0001234 Phenotype name", ...],
  "queries": [
    {"angle": "broad", "query": "..."},
    {"angle": "specific", "query": "..."},
    {"angle": "genetic", "query": "..."},
    {"angle": "system-based", "query": "..."},
    {"angle": "temporal", "query": "..."}
  ]
}
