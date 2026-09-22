"""Tests for Baseline Models (Majority, TF-IDF, Rule-Based)."""

import pytest
from evaluation.baselines import MajorityClassBaseline, TfidfIntentClassifier, RuleBasedBaseline
from src.schemas import IntentClassificationResult


def test_rule_based_baseline_predictions():
    rb = RuleBasedBaseline()

    # Battery
    res = rb.predict("My phone battery is draining and charger is hot")
    assert res.intent == "battery_and_charging"
    assert res.confidence >= 0.45

    # Billing
    res_bill = rb.predict("I was charged on itunes.com/bill and need a refund")
    assert res_bill.intent == "app_store_billing_and_subscriptions"

    # Empty
    res_empty = rb.predict("   ")
    assert res_empty.intent == "other"


def test_rule_based_baseline_escalation_decisions():
    rb = RuleBasedBaseline()

    # Financial / sensitive -> Escalate
    res_bill = rb.predict("Charged unauthorized money on App Store")
    dec_bill = rb.predict_decision("Charged unauthorized money on App Store", res_bill)
    assert dec_bill == "ESCALATE_TO_HUMAN"

    # Legal threat -> Escalate
    res_sue = rb.predict("I will sue you in court")
    dec_sue = rb.predict_decision("I will sue you in court", res_sue)
    assert dec_sue == "ESCALATE_TO_HUMAN"

    # Routine battery with keywords -> Auto
    res_bat = rb.predict("My battery is charging slowly")
    dec_bat = rb.predict_decision("My battery is charging slowly", res_bat)
    assert dec_bat == "AUTO_HANDLE"


def test_majority_baseline_batch():
    labels = ["battery_and_charging", "ios_software_update_bugs", "battery_and_charging"]
    maj = MajorityClassBaseline().fit(labels)
    preds = maj.predict_batch(["test 1", "test 2"])
    assert len(preds) == 2
    assert all(p.intent == "battery_and_charging" for p in preds)
