"""Generate publication-quality figures for the manuscript."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"
FIG_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colour palette (colourblind-safe)
BLUE = '#4477AA'
RED = '#EE6677'
GREEN = '#228833'
YELLOW = '#CCBB44'
GREY = '#BBBBBB'
PURPLE = '#AA3377'
CYAN = '#66CCEE'


def fig2_error_analysis():
    """Figure 2: Error analysis bar chart."""
    stats = json.load(open(BENCHMARK_DIR / "statistical_analyses.json"))
    cats = stats["failure_categories"]

    labels = ['RAG success', 'Retrieval\nfailure', 'Generation\nfailure', 'Knowledge\ngap']
    counts = [len(cats['rag_success']), len(cats['retrieval_failure']),
              len(cats['generation_failure']), len(cats['knowledge_gap'])]
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    colours = [GREEN, RED, YELLOW, GREY]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, pcts, color=colours, edgecolor='white', linewidth=0.5, width=0.65)

    for bar, count, pct in zip(bars, counts, pcts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{pct:.1f}%\n(n={count})', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Percentage of ultra-rare cases')
    ax.set_ylim(0, 65)
    ax.set_title('Failure mode analysis (n = 40 ultra-rare cases)', fontsize=11, pad=10)

    fig.savefig(FIG_DIR / 'fig2_error_analysis.pdf')
    fig.savefig(FIG_DIR / 'fig2_error_analysis.png')
    plt.close(fig)
    print("  Saved Figure 2")


def fig3_quadrant():
    """Figure 3: 4-quadrant retrieval vs generation decomposition."""
    stats = json.load(open(BENCHMARK_DIR / "statistical_analyses.json"))
    q = stats["quadrants"]

    data = np.array([
        [len(q['retrieved_correct']), len(q['retrieved_missed'])],
        [len(q['not_retrieved_correct']), len(q['not_retrieved_missed'])],
    ])
    total = data.sum()
    pct = data / total * 100

    fig, ax = plt.subplots(figsize=(5, 4))

    colours = np.array([
        [GREEN, YELLOW],
        [CYAN, RED],
    ])

    for i in range(2):
        for j in range(2):
            rect = plt.Rectangle((j, 1-i), 1, 1, facecolor=colours[i][j], alpha=0.7, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            ax.text(j + 0.5, 1.5 - i, f'{data[i][j]}\n({pct[i][j]:.1f}%)',
                    ha='center', va='center', fontsize=13, fontweight='bold')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['Generated\ncorrect', 'Generated\nwrong'], fontsize=10)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['Not\nretrieved', 'Retrieved'], fontsize=10)
    ax.set_xlabel('Diagnostic generation outcome', fontsize=11, labelpad=10)
    ax.set_ylabel('Retrieval outcome (top-20)', fontsize=11, labelpad=10)
    ax.set_title('Retrieval vs generation decomposition\n(n = 40 ultra-rare cases)', fontsize=11, pad=10)

    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(length=0)

    fig.savefig(FIG_DIR / 'fig3_quadrant.pdf')
    fig.savefig(FIG_DIR / 'fig3_quadrant.png')
    plt.close(fig)
    print("  Saved Figure 3")


def fig4_calibration():
    """Figure 4: Confidence calibration."""
    # Data from stability_runs.json (run 1)
    runs = json.load(open(BENCHMARK_DIR / "stability_runs.json"))
    run1 = runs["runs"][0]

    ur = [r for r in run1 if r["rarity"] == "ultra-rare"]

    # Count by confidence
    conf_data = {}
    for r in ur:
        conf = r.get("top1_confidence", "unknown")
        if conf not in conf_data:
            conf_data[conf] = {"total": 0, "correct": 0}
        conf_data[conf]["total"] += 1
        if r["correct_rank"] is not None and r["correct_rank"] <= 1:
            conf_data[conf]["correct"] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: distribution of confidence labels
    labels = ['High', 'Medium', 'Low', 'Other']
    counts = [
        conf_data.get('high', {}).get('total', 0),
        conf_data.get('medium', {}).get('total', 0),
        conf_data.get('low', {}).get('total', 0),
        sum(v['total'] for k, v in conf_data.items() if k not in ['high', 'medium', 'low']),
    ]
    ax1.bar(labels, counts, color=[RED, YELLOW, GREEN, GREY], edgecolor='white', width=0.6)
    ax1.set_ylabel('Number of predictions')
    ax1.set_title('Confidence label distribution', fontsize=10)
    for i, c in enumerate(counts):
        if c > 0:
            ax1.text(i, c + 0.5, str(c), ha='center', fontsize=9)

    # Right: accuracy by confidence
    conf_labels = ['High', 'Medium']
    accs = []
    for conf in ['high', 'medium']:
        d = conf_data.get(conf, {'total': 0, 'correct': 0})
        accs.append(d['correct'] / d['total'] * 100 if d['total'] > 0 else 0)

    bars = ax2.bar(conf_labels, accs, color=[RED, YELLOW], edgecolor='white', width=0.5)
    ax2.axhline(y=50, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Top-1 accuracy (%)')
    ax2.set_title('Accuracy by confidence level', fontsize=10)
    ax2.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', fontsize=9)

    fig.suptitle('LLM confidence calibration (ultra-rare cases)', fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_calibration.pdf')
    fig.savefig(FIG_DIR / 'fig4_calibration.png')
    plt.close(fig)
    print("  Saved Figure 4")


def fig5_stratified():
    """Figure 5: RAG advantage by disease category."""
    combined = json.load(open(BENCHMARK_DIR / "expanded_benchmark_combined.json"))
    ur = [r for r in combined if r["rarity"] == "ultra-rare"]

    # Load case data for categorisation
    all_cases = []
    for fname in ["test_cases.json", "nejm_cases.json", "ultra_rare_cases.json", "ultra_rare_cases_new.json"]:
        fpath = BENCHMARK_DIR / fname
        if fpath.exists():
            with open(fpath) as f:
                all_cases.extend(json.load(f))
    case_lookup = {c.get("case_id", ""): c for c in all_cases}

    def categorise(case_id, target):
        case = case_lookup.get(case_id, {})
        tl = target.lower()
        feats = " ".join(case.get("key_discriminating_features", [])).lower()
        ct = tl + " " + feats
        if any(w in tl for w in ["deletion", "microdeletion", "monosomy", "trisomy", "chromosome"]):
            return "Chromosomal"
        if any(w in tl for w in ["cdg", "metabolism"]) or "metabol" in ct:
            return "Metabolic"
        if any(w in tl for w in ["muscular", "myopath", "myotonic", "dystrophy"]):
            return "Neuromuscular"
        if any(w in tl for w in ["dysplasia", "dysostosis", "skeletal", "osteogenesis"]):
            return "Skeletal"
        if any(w in ct for w in ["intellectual disability", "developmental delay", "microcephaly"]):
            return "Neurodevelopmental"
        if any(w in tl for w in ["autoimmune", "lupus", "pemphigus", "psoriasis"]):
            return "Autoimmune"
        if any(w in tl for w in ["pulmonary", "respiratory", "hemosiderosis", "alveolar"]):
            return "Pulmonary"
        return "Other"

    cat_data = {}
    for r in ur:
        cat = categorise(r["case_id"], r["target"])
        cat_data.setdefault(cat, []).append(r)

    cats = []
    deltas = []
    ns = []
    for cat, cases in sorted(cat_data.items()):
        n = len(cases)
        if n < 2:
            continue
        rag_t1 = sum(1 for r in cases if r["rag_hyde_rank"] is not None and r["rag_hyde_rank"] <= 1) / n
        norag_t1 = sum(1 for r in cases if r["norag_rank"] is not None and r["norag_rank"] <= 1) / n
        cats.append(f"{cat}\n(n={n})")
        deltas.append((rag_t1 - norag_t1) * 100)
        ns.append(n)

    # Sort by delta
    order = np.argsort(deltas)[::-1]
    cats = [cats[i] for i in order]
    deltas = [deltas[i] for i in order]

    fig, ax = plt.subplots(figsize=(6, 4))
    colours = [GREEN if d > 0 else (RED if d < 0 else GREY) for d in deltas]
    bars = ax.barh(range(len(cats)), deltas, color=colours, edgecolor='white', height=0.6)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats)
    ax.set_xlabel('RAG advantage (percentage points)')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_title('RAG+HyDE advantage over LLM-only\nby disease category (ultra-rare cases)', fontsize=11, pad=10)
    ax.invert_yaxis()

    for bar, d in zip(bars, deltas):
        x = bar.get_width()
        ax.text(x + (2 if x >= 0 else -2), bar.get_y() + bar.get_height()/2,
                f'{d:+.0f}pp', ha='left' if x >= 0 else 'right', va='center', fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_stratified.pdf')
    fig.savefig(FIG_DIR / 'fig5_stratified.png')
    plt.close(fig)
    print("  Saved Figure 5")


def main():
    print("Generating manuscript figures...")
    fig2_error_analysis()
    fig3_quadrant()
    fig4_calibration()
    fig5_stratified()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
