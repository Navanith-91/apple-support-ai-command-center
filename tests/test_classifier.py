"""Unit tests for Intent Classifiers (Majority, TF-IDF, Semantic, Hybrid)."""

import pytest
from src.schemas import IntentClassificationResult
from src.intent_classifier import SemanticEmbeddingClassifier, HybridIntentClassifier
from evaluation.baselines import MajorityClassBaseline, TfidfIntentClassifier


def test_majority_baseline():
    labels = ["battery_and_charging", "battery_and_charging", "ios_software_update_bugs"]
    baseline = MajorityClassBaseline().fit(labels)
    res = baseline.predict("My screen is cracked")
    assert isinstance(res, IntentClassificationResult)
    assert res.intent == "battery_and_charging"
    assert res.confidence == pytest.approx(2/3, 0.01)


def test_tfidf_classifier():
    texts = [
        "battery draining fast and overheating while charging",
        "iOS 11 update broke my keyboard autocorrect and apps crashing",
        "my apple id account is locked and cannot log in",
        "charged 9.99 on itunes for a refund"
    ]
    labels = [
        "battery_and_charging",
        "ios_software_update_bugs",
        "apple_id_and_icloud",
        "app_store_billing_and_subscriptions"
    ]
    clf = TfidfIntentClassifier(max_features=100)
    clf.fit(texts, labels)
    
    res = clf.predict("Why is my battery discharging quickly?")
    assert isinstance(res, IntentClassificationResult)
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.scores) == 4


def test_semantic_embedding_classifier():
    clf = SemanticEmbeddingClassifier()
    res = clf.predict("My iPhone battery dies in 2 hours and charger gets hot")
    assert isinstance(res, IntentClassificationResult)
    assert res.intent == "battery_and_charging"
    assert res.confidence > 0.40
    assert "battery" in res.reason.lower()


def test_hybrid_classifier():
    hybrid = HybridIntentClassifier()
    res = hybrid.predict("Need a refund for an accidental in-app purchase on App Store")
    assert isinstance(res, IntentClassificationResult)
    assert res.intent == "app_store_billing_and_subscriptions"
    assert res.confidence >= 0.50
