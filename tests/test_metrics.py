"""Unit tests for metric calculations."""

import pytest
from evaluation.metrics import compute_intent_metrics, compute_escalation_metrics


def test_compute_intent_metrics():
    y_true = ["battery_and_charging", "ios_software_update_bugs", "apple_id_and_icloud"]
    y_pred = ["battery_and_charging", "ios_software_update_bugs", "battery_and_charging"]
    
    metrics = compute_intent_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(2/3, 0.01)
    assert "macro_f1" in metrics
    assert "per_intent" in metrics


def test_compute_escalation_metrics():
    y_true = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN", "ESCALATE_TO_HUMAN", "AUTO_HANDLE"]
    y_pred = ["AUTO_HANDLE", "ESCALATE_TO_HUMAN", "AUTO_HANDLE", "AUTO_HANDLE"]
    
    metrics = compute_escalation_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert metrics["false_auto_handle_count"] == 1
    assert metrics["false_auto_handling_rate"] == 0.50
    assert metrics["unnecessary_escalation_count"] == 0
