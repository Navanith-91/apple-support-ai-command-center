"""Unit and integration tests for the Customer Support AI Agent."""

import pytest
from src.agent import CustomerSupportAgent
from src.schemas import AgentResponse


def test_agent_end_to_end():
    agent = CustomerSupportAgent()
    resp = agent.run("My iPhone 7 battery percentage jumps from 50% to 10% randomly.")
    
    assert isinstance(resp, AgentResponse)
    assert resp.intent in ["battery_and_charging", "ios_software_update_bugs"]
    assert 0.0 <= resp.confidence <= 1.0
    assert resp.decision in ["AUTO_HANDLE", "ESCALATE_TO_HUMAN"]
    assert len(resp.reply) > 10
    assert "total_latency_ms" in resp.metadata


def test_agent_empty_message():
    agent = CustomerSupportAgent()
    resp = agent.run("   ")
    assert resp.intent == "other"
    assert resp.confidence == 0.0
    assert len(resp.reply) > 0


def test_agent_escalation_flow():
    agent = CustomerSupportAgent()
    resp = agent.run("Unauthorized charge on my credit card! I demand a refund right now!")
    assert resp.decision == "ESCALATE_TO_HUMAN"
    assert resp.intent == "app_store_billing_and_subscriptions"
