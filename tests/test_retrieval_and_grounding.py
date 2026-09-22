"""Tests for retrieval quality, evidence sufficiency, and response grounding."""

import pytest
import pandas as pd
from src.retriever import HistoricalRetriever
from src.response_generator import GroundedResponseGenerator
from src.schemas import RetrievedEvidence


def test_evidence_sufficiency_check():
    retriever = HistoricalRetriever(similarity_threshold=0.45)
    
    strong_ev = [RetrievedEvidence(
        conversation_id="1", score=0.75, customer_query="q", agent_reply="r"
    )]
    assert retriever.is_evidence_sufficient(strong_ev) is True

    weak_ev = [RetrievedEvidence(
        conversation_id="2", score=0.35, customer_query="q", agent_reply="r"
    )]
    assert retriever.is_evidence_sufficient(weak_ev) is False
    assert retriever.is_evidence_sufficient([]) is False


def test_retrieval_benchmark_evaluation():
    df = pd.DataFrame([
        {
            "conversation_id": "c1",
            "customer_text_clean": "My battery is draining fast",
            "agent_text_clean": "Please check Settings > Battery.",
            "brand": "AppleSupport",
            "created_at_customer": "2017-10-01",
            "intent": "battery_and_charging"
        }
    ])
    retriever = HistoricalRetriever(top_k=1, similarity_threshold=0.40)
    retriever.build_index(df)

    metrics = retriever.evaluate_retrieval_benchmarks(["battery is draining fast", "random unindexed topic xyz"])
    assert "mean_top1_similarity" in metrics
    assert "evidence_sufficiency_rate" in metrics
    assert metrics["evaluated_queries"] == 2


def test_grounded_response_generator_weak_evidence_fallback():
    generator = GroundedResponseGenerator()
    
    # Weak evidence (< 0.40) should trigger clarification fallback and mark grounded=False
    weak_ev = [RetrievedEvidence(
        conversation_id="1", score=0.30, customer_query="unrelated query", agent_reply="unrelated reply"
    )]
    res = generator.generate_response("obscure hardware issue", "other", weak_ev)
    assert res.grounded is False
    assert res.uncertainty is not None
    assert "device model" in res.reply.lower() or "software version" in res.reply.lower()


def test_grounded_response_generator_strong_evidence():
    generator = GroundedResponseGenerator()
    
    strong_ev = [RetrievedEvidence(
        conversation_id="1", score=0.82, customer_query="battery life", agent_reply="Let's look into this battery issue. Check Settings > Battery."
    )]
    res = generator.generate_response("My battery life is bad", "battery_and_charging", strong_ev)
    assert res.grounded is True
    assert "Settings > Battery" in res.reply
    assert len(res.safety_notes) > 0
