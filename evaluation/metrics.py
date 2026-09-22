"""Comprehensive evaluation metrics for Intent Classification and Escalation Decisions."""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


def _normalize_decision(d: str) -> str:
    """Normalizes AUTO / AUTO_HANDLE and ESCALATE / ESCALATE_TO_HUMAN."""
    d_clean = str(d).strip().upper()
    if d_clean in ("AUTO", "AUTO_HANDLE"):
        return "AUTO_HANDLE"
    if d_clean in ("ESCALATE", "ESCALATE_TO_HUMAN"):
        return "ESCALATE_TO_HUMAN"
    return d_clean


def compute_intent_metrics(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    """Computes overall and per-intent classification metrics."""
    all_labels = labels or sorted(list(set(y_true).union(set(y_pred))))
    
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )

    per_intent_df = pd.DataFrame({
        "intent": all_labels,
        "precision": [round(float(p), 4) for p in per_class_p],
        "recall": [round(float(r), 4) for r in per_class_r],
        "f1_score": [round(float(f), 4) for f in per_class_f1],
        "support": [int(s) for s in per_class_sup]
    })

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)

    return {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(prec_macro), 4),
        "macro_recall": round(float(rec_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "per_intent": per_intent_df,
        "confusion_matrix": cm,
        "labels": all_labels
    }


def compute_escalation_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes escalation metrics including safety-critical False Auto-Handling Rate."""
    y_true_clean = [_normalize_decision(y) for y in y_true]
    y_pred_clean = [_normalize_decision(y) for y in y_pred]

    acc = accuracy_score(y_true_clean, y_pred_clean)
    
    # Specific counts for safety analysis:
    # TP: Gold = ESCALATE_TO_HUMAN, Pred = ESCALATE_TO_HUMAN
    # FP: Gold = AUTO_HANDLE, Pred = ESCALATE_TO_HUMAN (Unnecessary Escalation)
    # FN: Gold = ESCALATE_TO_HUMAN, Pred = AUTO_HANDLE (False Auto-Handle - DANGEROUS)
    # TN: Gold = AUTO_HANDLE, Pred = AUTO_HANDLE
    
    total_gold_escalate = sum(1 for y in y_true_clean if y == "ESCALATE_TO_HUMAN")
    total_gold_auto = sum(1 for y in y_true_clean if y == "AUTO_HANDLE")

    false_auto_handle_count = sum(
        1 for yt, yp in zip(y_true_clean, y_pred_clean) 
        if yt == "ESCALATE_TO_HUMAN" and yp == "AUTO_HANDLE"
    )
    unnecessary_escalate_count = sum(
        1 for yt, yp in zip(y_true_clean, y_pred_clean) 
        if yt == "AUTO_HANDLE" and yp == "ESCALATE_TO_HUMAN"
    )

    # Rates
    false_auto_handle_rate = (false_auto_handle_count / total_gold_escalate) if total_gold_escalate > 0 else 0.0
    unnecessary_escalation_rate = (unnecessary_escalate_count / total_gold_auto) if total_gold_auto > 0 else 0.0

    # Binary metrics targeting ESCALATE_TO_HUMAN as positive class
    p, r, f1, _ = precision_recall_fscore_support(
        y_true_clean, y_pred_clean, pos_label="ESCALATE_TO_HUMAN", average="binary", zero_division=0
    )

    return {
        "accuracy": round(float(acc), 4),
        "escalation_precision": round(float(p), 4),
        "escalation_recall": round(float(r), 4),
        "escalation_f1": round(float(f1), 4),
        "false_auto_handle_count": false_auto_handle_count,
        "false_auto_handling_rate": round(float(false_auto_handle_rate), 4),
        "unnecessary_escalation_count": unnecessary_escalate_count,
        "unnecessary_escalation_rate": round(float(unnecessary_escalation_rate), 4),
        "total_gold_escalate": total_gold_escalate,
        "total_gold_auto": total_gold_auto,
        "total_evaluated": len(y_true)
    }


def format_metrics_table(intent_metrics: Dict[str, Any], escalation_metrics: Dict[str, Any]) -> str:
    """Formats combined summary table in Markdown."""
    lines = [
        "### Benchmark Evaluation Results Summary\n",
        "| Metric | Value | Interpretation / Target |",
        "| :--- | :--- | :--- |",
        f"| **Intent Accuracy** | `{intent_metrics['accuracy']:.2%}` | Primary multi-class accuracy across 9 intents |",
        f"| **Intent Macro F1** | `{intent_metrics['macro_f1']:.4f}` | Balanced performance across all intent classes |",
        f"| **Intent Weighted F1** | `{intent_metrics['weighted_f1']:.4f}` | Support-weighted intent F1 |",
        f"| **Escalation Precision** | `{escalation_metrics['escalation_precision']:.2%}` | Precision when recommending human handoff |",
        f"| **Escalation Recall** | `{escalation_metrics['escalation_recall']:.2%}` | Sensitivity in catching high-risk cases |",
        f"| **Escalation F1** | `{escalation_metrics['escalation_f1']:.4f}` | Harmonic mean of escalation decisions |",
        f"| **False Auto-Handling Rate** | `{escalation_metrics['false_auto_handling_rate']:.2%}` | **Safety Critical** (Target: < 5%) |",
        f"| **Unnecessary Escalation Rate** | `{escalation_metrics['unnecessary_escalation_rate']:.2%}` | Operational efficiency (Target: < 15%) |"
    ]
    return "\n".join(lines)
