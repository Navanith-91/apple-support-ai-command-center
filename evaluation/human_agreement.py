"""Phase 5: Human vs LLM Judge Agreement Study and Statistical Reliability Analysis.

Compares genuine, independently collected human ratings (evaluation/human_ratings.csv)
against LLM Judge scores. If human ratings are not yet collected, it reports a clear
'pending' status without fabricating or duplicating scores.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


def compute_judge_agreement_statistics(
    human_scores: List[float],
    llm_scores: List[float]
) -> Dict[str, Any]:
    """Computes exact agreement, MAD, Pearson r, Spearman rho, and quadratic weighted Kappa."""
    if len(human_scores) != len(llm_scores):
        raise ValueError(f"Score lists must have the same length (got human={len(human_scores)}, llm={len(llm_scores)})")

    if not human_scores or len(human_scores) == 0:
        return {
            "status": "pending",
            "sample_size": 0,
            "message": "Human agreement evaluation pending. Run scripts/collect_human_ratings.py to collect human ratings.",
            "exact_agreement": None,
            "adjacent_agreement_plus_minus_1": None,
            "mean_absolute_difference": None,
            "pearson_correlation": None,
            "pearson_p_value": None,
            "spearman_rank_correlation": None,
            "quadratic_weighted_kappa": None,
            "mean_human_score": None,
            "mean_llm_score": None
        }

    h_arr = np.array(human_scores, dtype=float)
    l_arr = np.array(llm_scores, dtype=float)
    n = len(h_arr)

    exact_match = float(np.sum(np.round(h_arr) == np.round(l_arr)) / n)
    within_one = float(np.sum(np.abs(np.round(h_arr) - np.round(l_arr)) <= 1) / n)
    mad = float(np.mean(np.abs(h_arr - l_arr)))

    mean_h = float(np.mean(h_arr))
    mean_l = float(np.mean(l_arr))

    # Pearson & Spearman correlation (requires at least 2 points and non-zero variance)
    if n >= 2 and np.std(h_arr) > 1e-6 and np.std(l_arr) > 1e-6:
        p_corr, p_val = pearsonr(h_arr, l_arr)
        s_corr, s_val = spearmanr(h_arr, l_arr)
    else:
        p_corr, p_val = (1.0, 0.0) if np.array_equal(h_arr, l_arr) else (0.0, 1.0)
        s_corr, s_val = (1.0, 0.0) if np.array_equal(h_arr, l_arr) else (0.0, 1.0)

    # Quadratic weighted Cohen's Kappa for ordinal ratings (1-5 discrete)
    h_discrete = np.clip(np.round(h_arr), 1, 5).astype(int)
    l_discrete = np.clip(np.round(l_arr), 1, 5).astype(int)
    try:
        if len(set(h_discrete).union(set(l_discrete))) > 1:
            kappa = cohen_kappa_score(h_discrete, l_discrete, weights="quadratic")
        else:
            kappa = 1.0
    except Exception:
        kappa = 0.0

    return {
        "status": "completed",
        "sample_size": n,
        "mean_human_score": round(mean_h, 2),
        "mean_llm_score": round(mean_l, 2),
        "exact_agreement": round(exact_match, 4),
        "adjacent_agreement_plus_minus_1": round(within_one, 4),
        "mean_absolute_difference": round(mad, 4),
        "pearson_correlation": round(float(p_corr), 4),
        "pearson_p_value": float(p_val),
        "spearman_rank_correlation": round(float(s_corr), 4),
        "quadratic_weighted_kappa": round(float(kappa), 4)
    }


def evaluate_human_agreement_from_file(
    ratings_csv_path: Path,
    llm_scores_map: Dict[str, float]
) -> Dict[str, Any]:
    """Reads human ratings CSV and compares with paired LLM judge scores."""
    if not ratings_csv_path.exists() or ratings_csv_path.stat().st_size == 0:
        return {
            "status": "pending",
            "sample_size": 0,
            "message": "Human agreement evaluation pending. Collect independent human ratings using scripts/collect_human_ratings.py."
        }

    df = pd.read_csv(ratings_csv_path, dtype=str).fillna("")
    valid_human_scores = []
    valid_llm_scores = []

    for _, row in df.iterrows():
        ex_id = str(row.get("example_id", "")).strip()
        h_ov = str(row.get("human_overall", "")).strip()
        if not h_ov:
            continue
        try:
            h_val = float(h_ov)
            if ex_id in llm_scores_map:
                l_val = float(llm_scores_map[ex_id])
                valid_human_scores.append(h_val)
                valid_llm_scores.append(l_val)
        except ValueError:
            continue

    if not valid_human_scores:
        return {
            "status": "pending",
            "sample_size": 0,
            "message": "Human agreement evaluation pending. No completed ratings found in evaluation/human_ratings.csv."
        }

    return compute_judge_agreement_statistics(valid_human_scores, valid_llm_scores)


def format_agreement_table(stats: Dict[str, Any]) -> str:
    """Formats human-LLM agreement analysis table for report."""
    if stats.get("status") == "pending" or stats.get("sample_size", 0) == 0:
        return (
            "### Human vs. LLM-as-a-Judge Agreement Study\n\n"
            "> [!NOTE]\n"
            "> **Status**: Human agreement evaluation pending.\n"
            "> Run `python scripts/collect_human_ratings.py` to collect at least 30 independent human ratings.\n"
        )

    lines = [
        "### Human vs. LLM-as-a-Judge Agreement Study\n",
        f"**Sample Size**: `{stats['sample_size']} independently rated customer interactions`\n",
        f"**Mean Human Rating**: `{stats['mean_human_score']} / 5.0` | **Mean LLM Judge Rating**: `{stats['mean_llm_score']} / 5.0`\n",
        "| Agreement Metric | Measured Value | Standard Interpretation |",
        "| :--- | :--- | :--- |",
        f"| **Exact Rating Agreement** | `{stats['exact_agreement']:.1%}` | Strict identical score (1-5) |",
        f"| **Adjacent Agreement ($\pm 1$)** | `{stats['adjacent_agreement_plus_minus_1']:.1%}` | Close agreement within 1 point |",
        f"| **Mean Absolute Difference (MAD)** | `{stats['mean_absolute_difference']:.2f}` | Average point deviation |",
        f"| **Pearson Correlation ($r$)** | `{stats['pearson_correlation']:.3f}` | Linear score correlation |",
        f"| **Spearman Rank Correlation ($\\rho$)** | `{stats['spearman_rank_correlation']:.3f}` | Monotonic ranking agreement |",
        f"| **Quadratic Weighted Kappa ($\\kappa$)** | `{stats['quadratic_weighted_kappa']:.3f}` | Inter-rater agreement reliability |"
    ]
    return "\n".join(lines)
