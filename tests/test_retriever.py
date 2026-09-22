"""Unit tests for Historical Semantic Retriever."""

import pytest
import pandas as pd
from src.retriever import HistoricalRetriever
from src.schemas import RetrievedEvidence


def test_retriever_build_and_query():
    df = pd.DataFrame([
        {
            "conversation_id": "c1",
            "customer_text_clean": "My iPhone 7 battery is draining fast",
            "agent_text_clean": "We'd like to help. Let's check Settings > Battery > Battery Health.",
            "brand": "AppleSupport",
            "created_at_customer": "2017-10-01",
            "intent": "battery_and_charging"
        },
        {
            "conversation_id": "c2",
            "customer_text_clean": "How do I reset my Apple ID password?",
            "agent_text_clean": "You can reset your Apple ID password at iforgot.apple.com.",
            "brand": "AppleSupport",
            "created_at_customer": "2017-10-02",
            "intent": "apple_id_and_icloud"
        }
    ])

    retriever = HistoricalRetriever(top_k=1, similarity_threshold=0.30)
    retriever.build_index(df)

    results = retriever.retrieve("iPhone battery life is terrible and dying quickly")
    assert len(results) == 1
    assert isinstance(results[0], RetrievedEvidence)
    assert results[0].conversation_id == "c1"
    assert results[0].score > 0.40
    assert "Settings > Battery" in results[0].agent_reply


def test_retriever_empty_query():
    retriever = HistoricalRetriever()
    assert retriever.retrieve("") == []
    assert retriever.retrieve("    ") == []
