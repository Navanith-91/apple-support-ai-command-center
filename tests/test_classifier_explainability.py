"""Tests for Intent Classifier explainability, confidence margins, and uncertainty detection."""

import pytest
from src.intent_classifier import SemanticEmbeddingClassifier, HybridIntentClassifier


def test_classifier_explainability_fields():
    clf = SemanticEmbeddingClassifier()
    res = clf.predict("My iPhone 7 battery is dying in 2 hours")

    assert hasattr(res, "confidence_margin")
    assert hasattr(res, "semantic_similarity")
    assert hasattr(res, "keyword_evidence")
    assert hasattr(res, "top_alternatives")
    assert hasattr(res, "is_uncertain")

    assert isinstance(res.top_alternatives, list)
    assert len(res.top_alternatives) > 0
    assert isinstance(res.confidence_margin, float)
    assert res.semantic_similarity > 0.0


def test_how_to_vs_software_bug_disambiguation():
    clf = SemanticEmbeddingClassifier()

    # Pure how-to question should predict general_how_to_and_features
    res_howto = clf.predict("How do I transfer photos from my old iPhone to my new iPhone 8?")
    assert res_howto.intent == "general_how_to_and_features"

    # Actual bug report should predict ios_software_update_bugs
    res_bug = clf.predict("Ever since updating to iOS 11 my keyboard freezes and crashes constantly")
    assert res_bug.intent == "ios_software_update_bugs"


def test_uncertainty_detection_on_ambiguous_input():
    clf = SemanticEmbeddingClassifier()
    res = clf.predict("it broke ????")
    # Low confidence or small margin flags uncertainty
    assert res.is_uncertain is True or res.confidence < 0.50
