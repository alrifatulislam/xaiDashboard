# metrics.py
from typing import List
import numpy as np
from scipy.stats import spearmanr
from rouge_score import rouge_scorer


# ---------- ROUGE-L BASELINE ---------- #

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def summary_rougeL(reference: str, generated: str) -> float:
    """
    ROUGE-L F1 between reference and generated summary.
    """
    reference = reference or ""
    generated = generated or ""
    scores = _rouge.score(reference, generated)
    return float(scores["rougeL"].fmeasure)


# ---------- BASIC SPEARMAN UTILITY ---------- #

def compute_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """
    Spearman rank correlation between two vectors.
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if a.size == 0 or b.size == 0 or len(a) != len(b):
        return float("nan")
    rho, _ = spearmanr(a, b)
    return float(rho)


# ---------- FAITHFULNESS VIA TOKEN DELETION ---------- #

def faithfulness_via_deletion(
    text: str,
    words: List[str],
    importances: np.ndarray,
    model,
    reference_summary: str,
    max_tokens: int = 25,
) -> float:
    """
    Faithfulness metric:

    For each word i (up to max_tokens):
      - delete that word from the text
      - recompute summary
      - measure drop in ROUGE-L vs base summary

    Then return Spearman correlation between:
      - explanation importances
      - drops in ROUGE-L

    Higher correlation = more faithful explanations.
    """
    words = list(words)
    importances = np.array(importances, dtype=float)

    if len(words) == 0 or importances.size == 0:
        return float("nan")

    n = min(len(words), max_tokens)
    words = words[:n]
    importances = importances[:n]

    base_summary, _ = model.summarize(text)
    base_score = summary_rougeL(reference_summary, base_summary)

    deltas = []
    for i in range(n):
        masked = " ".join(w for j, w in enumerate(words) if j != i)
        summary_i, _ = model.summarize(masked)
        score_i = summary_rougeL(reference_summary, summary_i)
        deltas.append(base_score - score_i)

    deltas = np.array(deltas, dtype=float)
    return compute_spearman(importances, deltas)
