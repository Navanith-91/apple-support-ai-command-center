"""Phase 28: Interactive multi-scenario demonstration script."""

import sys
import os
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.agent import CustomerSupportAgent

DEMO_SCENARIOS = [
    {
        "title": "Scenario 1: Routine Diagnostic & Auto-Handled Technical Issue",
        "category": "AUTO_HANDLE (Battery)",
        "message": "My iPhone 7 battery is draining from 100% to 15% in less than three hours after normal use. Why is this happening?"
    },
    {
        "title": "Scenario 2: Routine Feature / How-To Configuration",
        "category": "AUTO_HANDLE (How-To)",
        "message": "How do I transfer all my photos and contacts from an older iPhone to my new iPhone 8 using Quick Start?"
    },
    {
        "title": "Scenario 3: High-Risk Financial Dispute & Unauthorized Charge",
        "category": "ESCALATE_TO_HUMAN (Financial)",
        "message": "I see an unauthorized charge of $89.99 from itunes.com/bill on my credit card that I never made. I demand an immediate refund!"
    },
    {
        "title": "Scenario 4: Account Security & Two-Factor Authentication Lockout",
        "category": "ESCALATE_TO_HUMAN (Security)",
        "message": "My Apple ID has been locked for security reasons and I can't receive the two-factor verification code on my trusted phone number."
    },
    {
        "title": "Scenario 5: High Customer Frustration & Supervisor Escalation",
        "category": "ESCALATE_TO_HUMAN (Anger/Manager)",
        "message": "Your customer support is completely useless! I've been waiting for two weeks and nobody resolved my case. Get me a supervisor right now or I'm suing!"
    },
    {
        "title": "Scenario 6: Ambiguous / Low-Context Edge Case",
        "category": "ESCALATE_TO_HUMAN / Clarification (Ambiguity)",
        "message": "help please it broke ????"
    }
]


def run_demo():
    print("\n" + "="*80)
    print("   AI CUSTOMER SUPPORT AGENT -- MULTI-SCENARIO PRODUCTION DEMO")
    print("   Brand: AppleSupport | Grounding: Historical Twitter Support Dataset")
    print("="*80 + "\n")

    config = load_config()
    agent = CustomerSupportAgent(config=config)

    for idx, scenario in enumerate(DEMO_SCENARIOS, start=1):
        print(f"\n>>> [{idx}/6] {scenario['title']} ({scenario['category']})")
        print(f"Customer Message: \"{scenario['message']}\"")
        print("-" * 75)

        resp = agent.run(scenario["message"])

        print(f"PREDICTED INTENT:    {resp.intent} (Confidence: {resp.confidence:.2%})")
        print(f"DECISION:            {resp.decision}")
        print(f"REASON:              {resp.reason}")
        print("-" * 75)
        print("DRAFTED OFFICIAL REPLY:")
        print(f"\"{resp.reply}\"")
        print("-" * 75)
        print("RETRIEVED HISTORICAL EVIDENCE:")
        if resp.evidence:
            for e_idx, ev in enumerate(resp.evidence[:2], start=1):
                print(f"  [{e_idx}] Score: {ev.get('score', 0):.2f} | Query: {str(ev.get('customer_query', ''))[:50]}... | Reply: {str(ev.get('agent_reply', ''))[:60]}...")
        else:
            print("  No matching historical evidence found.")
        print(f"Execution Latency:   {resp.metadata.get('total_latency_ms', 0):.1f} ms")
        print("=" * 80)


if __name__ == "__main__":
    run_demo()
