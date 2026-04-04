"""Generate narrative clinical descriptions for all diseases.

Usage:
    python scripts/generate_narratives.py [--batch-size 50] [--model claude-haiku-4-5-20251001]

Supports resume — checks existing narratives.json and skips already-generated diseases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic
from src.ingest.orphanet_ingest import load_diseases
from src.index.narrative_generator import (
    generate_narratives_batch, load_narratives, save_narratives,
)


def main(model: str = "claude-haiku-4-5-20251001", batch_size: int = 50):
    print("=" * 60)
    print("Generating Narrative Clinical Descriptions")
    print(f"Model: {model}")
    print("=" * 60)

    # Load diseases
    diseases = load_diseases()
    print(f"Loaded {len(diseases)} diseases")

    # Load existing narratives (for resume)
    existing = load_narratives()
    print(f"Already generated: {len(existing)} narratives")

    # Filter to diseases that need narratives
    # Only generate for diseases with >= 5 HPO associations (worth narrating)
    to_generate = [
        d for d in diseases
        if d.orpha_code not in existing and len(d.hpo_associations) >= 5
    ]
    print(f"To generate: {len(to_generate)} diseases")

    if not to_generate:
        print("Nothing to generate!")
        return

    client = anthropic.Anthropic()

    # Process in batches with periodic saving
    all_narratives = dict(existing)
    total_batches = (len(to_generate) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(to_generate))
        batch = to_generate[start:end]

        print(f"\nBatch {batch_idx + 1}/{total_batches} ({start+1}-{end}/{len(to_generate)})")

        batch_results = generate_narratives_batch(
            diseases=batch,
            client=client,
            model=model,
            delay=0.15,
        )

        all_narratives.update(batch_results)

        # Save after each batch
        save_narratives(all_narratives)
        print(f"  Saved {len(all_narratives)} total narratives")

    print(f"\n{'='*60}")
    print(f"Done. Generated {len(all_narratives)} narratives total.")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    main(model=args.model, batch_size=args.batch_size)
