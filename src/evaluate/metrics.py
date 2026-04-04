"""Evaluation metrics for diagnostic accuracy."""

from typing import Optional


def normalize_disease_name(name: str) -> str:
    """Normalize a disease name for comparison."""
    import re
    name = name.lower().strip()
    # Remove common prefixes/suffixes
    for word in ["syndrome", "disease", "disorder"]:
        name = name.replace(word, "").strip()
    # Remove punctuation
    name = re.sub(r"[,;:\-/\(\)]", " ", name)
    # Collapse whitespace
    name = " ".join(name.split())
    return name


def disease_match(predicted: str, ground_truth: str, aliases: list[str] = None) -> bool:
    """Check if a predicted disease name matches the ground truth."""
    pred_norm = normalize_disease_name(predicted)
    gt_norm = normalize_disease_name(ground_truth)

    if _name_match(pred_norm, gt_norm):
        return True

    # Check aliases
    if aliases:
        for alias in aliases:
            alias_norm = normalize_disease_name(alias)
            if _name_match(pred_norm, alias_norm):
                return True

    return False


def _name_match(a: str, b: str) -> bool:
    """Check if two normalized disease names match."""
    # Exact match
    if a == b:
        return True

    # Substring match (either direction)
    if a in b or b in a:
        return True

    # High word overlap (>=70% of words in common)
    words_a = set(a.split())
    words_b = set(b.split())
    if words_a and words_b:
        overlap = len(words_a & words_b)
        shorter = min(len(words_a), len(words_b))
        if shorter > 0 and overlap / shorter >= 0.7:
            return True

    return False


def top_k_accuracy(
    predictions: list[list[str]],  # For each case: list of predicted disease names
    ground_truths: list[str],
    aliases: list[list[str]] = None,
    k: int = 1,
) -> float:
    """Proportion of cases where correct diagnosis is in top-k predictions."""
    if not predictions:
        return 0.0

    correct = 0
    for i, (preds, gt) in enumerate(zip(predictions, ground_truths)):
        case_aliases = aliases[i] if aliases else []
        top_preds = preds[:k]
        if any(disease_match(p, gt, case_aliases) for p in top_preds):
            correct += 1

    return correct / len(predictions)


def mean_reciprocal_rank(
    predictions: list[list[str]],
    ground_truths: list[str],
    aliases: list[list[str]] = None,
) -> float:
    """Average of 1/rank where rank is the position of the correct diagnosis."""
    if not predictions:
        return 0.0

    rr_sum = 0.0
    for i, (preds, gt) in enumerate(zip(predictions, ground_truths)):
        case_aliases = aliases[i] if aliases else []
        for rank, pred in enumerate(preds, start=1):
            if disease_match(pred, gt, case_aliases):
                rr_sum += 1.0 / rank
                break

    return rr_sum / len(predictions)


def retrieval_recall_at_k(
    retrieved_diseases: list[str],
    ground_truth_disease: str,
    aliases: list[str] = None,
    k: int = 20,
) -> bool:
    """Did the retriever find the correct disease in top-k results?"""
    for disease in retrieved_diseases[:k]:
        if disease_match(disease, ground_truth_disease, aliases):
            return True
    return False


def hallucination_rate(
    predicted_diseases: list[str],
    retrieved_diseases: set[str],
) -> float:
    """Proportion of predicted diagnoses NOT found in retrieved evidence."""
    if not predicted_diseases:
        return 0.0

    hallucinated = 0
    for pred in predicted_diseases:
        found = any(
            disease_match(pred, ret_d)
            for ret_d in retrieved_diseases
        )
        if not found:
            hallucinated += 1

    return hallucinated / len(predicted_diseases)
