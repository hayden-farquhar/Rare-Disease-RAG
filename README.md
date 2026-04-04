# RAG-Powered Clinical Decision Support for Rare Disease Diagnosis

Replication code and data for:

> **Retrieval-augmented generation improves diagnostic accuracy for ultra-rare diseases**
>
> Hayden Farquhar. *npj Digital Medicine* (submitted).

## Overview

This repository provides the complete code and benchmark data to replicate the study's findings. The system retrieves from a structured knowledge base of 4,293 Orphanet diseases and 1,832 PubMed case reports, then uses an LLM to generate ranked differential diagnoses with cited evidence.

**Key result:** On 70 ultra-rare disease cases (prevalence <1/100,000), RAG achieved 54.3% top-1 diagnostic accuracy vs 38.6% for the LLM alone (+15.7 pp, McNemar's exact p = 0.001).

## Repository structure

```
├── src/                    # Core modules
│   ├── ingest/             # Data ingestion (Orphanet, PubMed, HPO)
│   ├── index/              # Embedding, chunking, FAISS indexing
│   ├── retrieve/           # Multi-source retrieval, RRF, HyDE
│   ├── generate/           # LLM diagnostic generation
│   └── evaluate/           # Metrics and benchmarking
├── scripts/                # Executable scripts for each experiment
├── prompts/                # LLM prompt templates
├── data/
│   ├── benchmarks/         # Benchmark cases and results
│   ├── hpo/                # HPO ontology (download separately)
│   ├── raw/                # Raw data (download separately)
│   └── processed/          # FAISS indices (built during setup)
└── tests/                  # Unit tests
```

## Setup

### Prerequisites

- Python 3.12+
- macOS, Linux, or WSL (FAISS requires Unix-like environment)
- Anthropic API key with credit balance (~$15 to replicate all experiments)

### Installation

```bash
# Clone the repository
git clone https://github.com/hayden-farquhar/Rare-Disease-RAG.git
cd Rare-Disease-RAG

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Data downloads

The following data must be downloaded separately due to size and licensing:

1. **Orphanet XML data** — Download from [Orphadata](https://www.orphadata.com/):
   - Product 1 (disease definitions)
   - Product 4 (HPO associations)
   - Product 6 (gene associations)
   - Product 9 (age of onset)
   - Place in `data/raw/orphanet/`

2. **HPO ontology** — Download `hp.obo` from [HPO](https://hpo.jax.org/):
   ```bash
   curl -o data/hpo/hp.obo https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo
   ```

3. **PubMed case reports** — Fetched automatically:
   ```bash
   python scripts/build_pubmed_index.py
   ```

## Replication guide

### Step 1: Build the knowledge base

```bash
# Parse Orphanet XML and build FAISS indices
python scripts/build_all_indices.py

# Generate narrative descriptions (requires API key, ~$3)
python scripts/generate_narratives.py

# Rebuild indices with narratives
python scripts/build_all_indices.py

# Build PubMed case report index
python scripts/build_pubmed_index.py
```

### Step 2: Run the main benchmark (Table 2)

```bash
# 4-approach comparison on original 55 cases
python scripts/run_all_approaches.py

# HyDE benchmark on 55 cases
python scripts/run_hyde_benchmark.py

# Expanded benchmark on 30 new cases + combined metrics
python scripts/run_expanded_benchmark.py
```

### Step 3: Run ablation studies (Table 3)

```bash
python scripts/run_ablations.py
python scripts/run_ablations_remaining.py
```

### Step 4: Statistical analyses

```bash
# Error analysis, KB coverage, retrieval-generation decomposition
python scripts/statistical_analyses.py

# Comprehensive statistical validation (McNemar, Wilcoxon, permutation, effect sizes)
python scripts/statistical_validation.py
```

### Step 5: Confidence calibration and stability (3 runs)

```bash
python scripts/run_confidence_and_stability.py
```

### Step 6: Generate figures

```bash
pip install matplotlib
python scripts/generate_figures.py
```

## Benchmark data

The `data/benchmarks/` directory contains:

| File | Description |
|------|-------------|
| `test_cases.json` | 5 initial test cases (well-characterised rare diseases) |
| `nejm_cases.json` | 10 NEJM-style clinical problem-solving cases |
| `ultra_rare_cases.json` | 40 ultra-rare cases curated from PMC |
| `ultra_rare_cases_new.json` | 30 additional ultra-rare cases |
| `expanded_benchmark_combined.json` | Combined results for all 85 cases |
| `all_approaches_results.json` | 4-approach comparison (55 cases) |
| `ablation_results.json` | Component and source ablation results |
| `hyde_benchmark_results.json` | HyDE evaluation results |
| `stability_runs.json` | 3-run reproducibility data |
| `statistical_analyses.json` | Error analysis and decomposition |

## Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| anthropic | >=0.39 | Claude API client |
| sentence-transformers | >=3.0 | Embedding models |
| faiss-cpu | >=1.7 | Vector similarity search |
| scikit-learn | >=1.3 | TF-IDF sparse retrieval |
| numpy | >=1.24 | Numerical operations |
| scipy | >=1.11 | Statistical tests |
| pydantic | >=2.0 | Data validation |
| httpx | >=0.25 | HTTP client (PubMed API) |
| lxml | >=4.9 | XML parsing (Orphanet) |

## Citation

If you use this code or data, please cite:

```bibtex
@article{farquhar2026rag,
  title={Retrieval-augmented generation improves diagnostic accuracy for ultra-rare diseases},
  author={Farquhar, Hayden},
  journal={npj Digital Medicine},
  year={2026},
  note={submitted}
}
```

## Licence

This project is licensed under the MIT Licence. See [LICENCE](LICENCE) for details.

Orphanet data is subject to [Orphadata terms of use](https://www.orphadata.com/terms-of-use/). PubMed data is in the public domain.
