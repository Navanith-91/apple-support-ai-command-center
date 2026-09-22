"""Unit tests for the Escalation Policy Engine."""

import pytest
from src.escalation import EscalationPolicyEngine
from src.schemas import DecisionEnum, IntentClassificationResult, RetrievedEvidence


def test_escalation_financial_intent():
    engine = EscalationPolicyEngine()
    clf = IntentClassificationResult(
        intent="app_store_billing_and_subscriptions",
        confidence=0.95,
        reason="Matched refund keywords."
    )
    res = engine.evaluate("I was charged twice for Apple Music", clf, [])
    assert res.decision == DecisionEnum.ESCALATE_TO_HUMAN
    assert any("billing" in s or "financial" in s for s in res.risk_signals + [res.reason.lower()])


def test_escalation_account_security():
    engine = EscalationPolicyEngine()
    clf = IntentClassificationResult(
        intent="apple_id_and_icloud",
        confidence=0.88,
        reason="Matched iCloud."
    )
    res = engine.evaluate("My Apple ID is locked and I can't get past 2FA", clf, [])
    assert res.decision == DecisionEnum.ESCALATE_TO_HUMAN
    assert any("security" in s or "apple_id" in s for s in res.risk_signals + [res.reason.lower()])


def test_escalation_risk_keywords():
    engine = EscalationPolicyEngine()
    clf = IntentClassificationResult(
        intent="hardware_audio_and_screen",
        confidence=0.85,
        reason="Screen repair."
    )
    res = engine.evaluate("My screen cracked and I demand to speak with your manager or I will sue", clf, [])
    assert res.decision == DecisionEnum.ESCALATE_TO_HUMAN
    assert any("manager" in s or "sue" in s for s in res.risk_signals + [res.reason.lower()])


def test_auto_handle_routine_issue():
    engine = EscalationPolicyEngine()
    clf = IntentClassificationResult(
        intent="battery_and_charging",
        confidence=0.92,
        reason="Matched battery drain."
    )
    ev = [RetrievedEvidence(
        conversation_id="conv_1",
        score=0.85,
        customer_query="battery dies quickly",
        agent_reply="Check Settings > Battery"
    )]
    res = engine.evaluate("How can I improve my battery life on iOS 11?", clf, ev)
    assert res.decision == DecisionEnum.AUTO_HANDLE
    assert "Routine" in res.reason and "battery_and_charging" in res.reason
