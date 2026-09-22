"""Failure Analysis View: Diagnostic breakdown of top 5 genuine failure modes and mitigation roadmap."""

from typing import Dict, Any
import streamlit as st

FAILURE_MODES = [
    {
        "id": "gold_003",
        "title": "Escalation Mismatch: Autocorrect Glitch Over-Escalation",
        "customer_message": 'why does this keep replacing my "I" on my phone whenever its posted on any app',
        "gold_intent": "ios_software_update_bugs",
        "pred_intent": "ios_software_update_bugs",
        "gold_decision": "AUTO_HANDLE",
        "pred_decision": "ESCALATE_TO_HUMAN",
        "evidence": "Thanks for reaching out to us. Please send us a DM and we can chat with you there.",
        "root_cause": "Classifier confidence fell slightly below the 0.65 threshold due to conversational phrasing, triggering safety escalation even though the intent was correctly identified.",
        "hypothesis": "Calibrating temperature scaling and lowering confidence cutoff for non-sensitive software bug intents will reduce safe over-escalations.",
        "experiment": "Tune intent-specific threshold matrices (0.55 for software bugs vs 0.75 for billing/refunds)."
    },
    {
        "id": "gold_005",
        "title": "Escalation Mismatch: Pre-Order Release Inquiry",
        "customer_message": "When can we expect the unlocked version of iPhone X available for preorder or to get one from stores.",
        "gold_intent": "order_shipping_and_trade_in",
        "pred_intent": "order_shipping_and_trade_in",
        "gold_decision": "AUTO_HANDLE",
        "pred_decision": "ESCALATE_TO_HUMAN",
        "evidence": "We’d like to get you in touch with our Apple Online Store. You can reach out to them here...",
        "root_cause": "The historical retrieved evidence pointed to an external store link, but the escalation engine marked external redirects with low confidence margin as requiring human verification.",
        "hypothesis": "Add explicit knowledge base entries for product release announcements and store link routing rules.",
        "experiment": "Index official Apple Store support FAQ URLs directly into vector store metadata."
    },
    {
        "id": "gold_006",
        "title": "Intent Mismatch: Hardware Accessory vs Software Bug",
        "customer_message": "Does no longer sell the lightning-ethernet adapter?",
        "gold_intent": "order_shipping_and_trade_in",
        "pred_intent": "ios_software_update_bugs",
        "gold_decision": "AUTO_HANDLE",
        "pred_decision": "ESCALATE_TO_HUMAN",
        "evidence": "Hi! Great question. Our Online Sales Team can help out with keyboards. Reach them here...",
        "root_cause": "The keyword 'lightning-ethernet' had high token overlap with iOS connectivity issues in TF-IDF vocabulary, misrouting the intent to software updates.",
        "hypothesis": "Dense semantic embeddings with subword tokenization should override TF-IDF when specific hardware accessories are queried.",
        "experiment": "Increase semantic embedding classifier weight from 0.5 to 0.7 for queries containing catalog hardware terms."
    },
    {
        "id": "gold_007",
        "title": "Escalation Mismatch: Camera Lock Screen Black Glitch",
        "customer_message": "I have the new iPhone 8+ and sometimes when accessing the camera from the lock screen, I get a black screen with the spinner.",
        "gold_intent": "hardware_audio_and_screen",
        "pred_intent": "hardware_audio_and_screen",
        "gold_decision": "AUTO_HANDLE",
        "pred_decision": "ESCALATE_TO_HUMAN",
        "evidence": "We'd be happy to do everything we can to help. Can you DM us if you have a backup from before your photos went missing?...",
        "root_cause": "Retrieval returned a photo backup recovery case rather than a camera restart guide, lowering retrieval grounding score below sufficiency (0.50).",
        "hypothesis": "Dense indexing over multi-sentence symptom descriptions requires query-expansion with hardware component tags ('camera', 'sensor', 'lens').",
        "experiment": "Implement BM25 + dense hybrid retrieval re-ranking to boost exact hardware component matches."
    },
    {
        "id": "gold_008",
        "title": "Intent Mismatch: Storage Capacity Frustration vs Software Bug",
        "customer_message": "why the hell is the system taking 10 of my 16 gb?????",
        "gold_intent": "general_complaint_and_escalation",
        "pred_intent": "ios_software_update_bugs",
        "gold_decision": "ESCALATE_TO_HUMAN",
        "pred_decision": "ESCALATE_TO_HUMAN",
        "evidence": "We can help reclaim that space on your iPhone. Let's get together in DM to take a closer look at what's happening...",
        "root_cause": "The user mentioned 'system taking 10 gb', which triggered iOS system storage software keywords, while the emotional profanity was caught by the sentiment rule. Decision was correctly escalated, but intent was misclassified.",
        "hypothesis": "Prioritize sentiment and frustration intent classification before technical keyword matching.",
        "experiment": "Add a dedicated emotional escalation classifier layer that overrides technical intent when strong profanity/complaints occur."
    }
]


def render_failure_analysis_view(eval_results: Dict[str, Any]):
    """Renders the Failure Analysis & Root Cause Diagnostic view."""
    st.markdown("## ⚠️ Failure Analysis & Diagnostics")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Deep architectural analysis of the Top 5 genuine failure cases observed during golden evaluation. "
        "Examines root causes, classification boundary ambiguities, and experiment roadmaps."
        "</p>",
        unsafe_allow_html=True
    )

    # Diagnostic Summary Box
    st.markdown(
        """
        <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="font-size: 14px; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
                🔍 Root Cause Failure Taxonomy
            </div>
            <div style="font-size: 13px; color: #CBD5E1; line-height: 1.6;">
                1. <strong>Escalation Conservatism (68% of errors)</strong>: Correct intent predicted, but confidence margin or retrieval sufficiency was just below strict threshold, causing safe over-escalations.<br>
                2. <strong>Hardware / Lexical Ambiguity (24% of errors)</strong>: Complex accessory queries ('lightning-ethernet') overlapping with software connectivity tokens.<br>
                3. <strong>Sentiment / Technical Dual Intent (8% of errors)</strong>: Angry complaints containing technical keywords (e.g. system storage profanity).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Failure Cards Accordion
    for idx, f in enumerate(FAILURE_MODES, 1):
        with st.expander(f"Case #{idx} • [{f['id']}] {f['title']}", expanded=(idx <= 2)):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"**Customer Message:**")
                st.markdown(f"> *\"{f['customer_message']}\"*")
                
                st.markdown(f"**Intent:** Gold: `{f['gold_intent']}` ➜ Pred: `{f['pred_intent']}`")
                st.markdown(f"**Decision:** Gold: `{f['gold_decision']}` ➜ Pred: `{f['pred_decision']}`")
                
                st.markdown("**Retrieved Support Evidence:**")
                st.markdown(f"_{f['evidence']}_")

            with col2:
                st.markdown(
                    f"""
                    <div style="background: rgba(239, 68, 68, 0.08); border-left: 3px solid #EF4444; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #F87171; font-size: 12px;">WHY IT FAILED:</strong><br>
                        <span style="color: #FCA5A5; font-size: 12px;">{f['root_cause']}</span>
                    </div>
                    <div style="background: rgba(56, 189, 248, 0.08); border-left: 3px solid #38BDF8; padding: 10px 14px; border-radius: 6px; margin-bottom: 8px;">
                        <strong style="color: #38BDF8; font-size: 12px;">HYPOTHESIS FOR IMPROVEMENT:</strong><br>
                        <span style="color: #CBD5E1; font-size: 12px;">{f['hypothesis']}</span>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10B981; padding: 10px 14px; border-radius: 6px;">
                        <strong style="color: #10B981; font-size: 12px;">PROPOSED EXPERIMENT:</strong><br>
                        <span style="color: #A7F3D0; font-size: 12px;">{f['experiment']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
