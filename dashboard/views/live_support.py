"""Live Support Console View: Real-time agent inference, preset scenario pills, and deep explainability."""

import time
import html
from typing import Dict, Any, Optional
import streamlit as st

from src.agent import CustomerSupportAgent
from dashboard.components.chat_bubble import render_customer_message, render_agent_message, render_escalation_notice
from dashboard.components.explainability import (
    render_decision_badge,
    render_risk_signals,
    render_alternative_intents_chart,
    render_retrieved_evidence,
    render_latency_breakdown
)
from dashboard.utils.formatters import format_intent_name, format_percentage, format_latency

# Preset test scenarios spanning standard and high-risk cases
PRESET_SCENARIOS = [
    {
        "label": "🔋 Battery Drain (iOS 11)",
        "text": "My iPhone battery is draining 50% in an hour after the iOS 11 update, please fix this!",
        "type": "standard"
    },
    {
        "label": "💳 App Store Billing Refund",
        "text": "I got charged $9.99 for an app subscription I cancelled last month, I demand an immediate refund.",
        "type": "risk"
    },
    {
        "label": "🔐 iCloud Lockout & 2FA",
        "text": "My Apple ID is locked and I cannot receive the two-factor authentication code because my device is disabled.",
        "type": "standard"
    },
    {
        "label": "📷 Camera Black Screen Glitch",
        "text": "When I open the camera app on my iPhone 8 Plus, it just shows a black screen with a spinner.",
        "type": "standard"
    },
    {
        "label": "📦 Order Shipping Status",
        "text": "When will my iPhone X order be delivered? The tracking link has not updated in 3 days.",
        "type": "standard"
    },
    {
        "label": "⚖️ Legal / Angry Escalation",
        "text": "Your customer service is horrific. I've waited 3 weeks and I am contacting my lawyer and the BBB tomorrow if this isn't resolved.",
        "type": "high_risk"
    }
]


def render_live_support_view(agent: CustomerSupportAgent):
    """Renders the interactive AI Support Console."""
    st.markdown("## 💬 Live AI Support Console")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Test real-time inquiries against the production AppleSupport AI agent. "
        "Inspect intent classification, semantic retrieval grounding, safety gates, and response synthesis."
        "</p>",
        unsafe_allow_html=True
    )

    # Preset Scenario Selector Pills
    st.markdown("##### ⚡ Quick Test Scenarios")
    pill_cols = st.columns(len(PRESET_SCENARIOS))
    
    # Initialize prompt state if not present
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = PRESET_SCENARIOS[0]["text"]

    for idx, (col, scenario) in enumerate(zip(pill_cols, PRESET_SCENARIOS)):
        with col:
            if st.button(scenario["label"], key=f"pill_{idx}", use_container_width=True):
                st.session_state.current_prompt = scenario["text"]
                st.session_state.auto_run = True

    # Customer Inbound Message Input Box
    st.markdown("##### 📥 Inbound Customer Inquiry")
    user_input = st.text_area(
        label="Customer Message",
        value=st.session_state.current_prompt,
        height=100,
        label_visibility="collapsed",
        placeholder="Type a customer support message to test the AI Agent in real-time..."
    )

    # Action Buttons Row
    btn_col1, btn_col2, btn_spacer = st.columns([2, 2, 6])
    with btn_col1:
        run_clicked = st.button("🚀 Run AI Agent", type="primary", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🔄 Clear", use_container_width=True)
        if clear_clicked:
            st.session_state.current_prompt = ""
            st.rerun()

    # Check if run triggered via button or pill click
    should_run = run_clicked or st.session_state.get("auto_run", False)
    if st.session_state.get("auto_run", False):
        st.session_state.auto_run = False

    if should_run and user_input.strip():
        with st.spinner("🤖 Running AI Agent Pipeline (Intent -> Vector RAG -> Escalation -> Generation)..."):
            t0 = time.perf_counter()
            response = agent.run(user_input.strip())
            elapsed_ms = (time.perf_counter() - t0) * 1000

        # Save to session activity stream
        if "activity_log" not in st.session_state:
            st.session_state.activity_log = []
        
        st.session_state.activity_log.insert(0, {
            "customer_message": user_input.strip(),
            "intent": response.intent,
            "decision": response.decision,
            "reason": response.reason,
            "confidence": response.confidence,
            "latency_ms": response.metadata.get("total_latency_ms", elapsed_ms),
            "reply": response.reply,
            "evidence": response.evidence,
            "metadata": response.metadata
        })

        st.markdown("---")
        st.markdown("### 🎯 Live Agent Execution Results")

        # Top Decision & Latency Row
        render_decision_badge(
            decision=response.decision,
            reason=response.reason,
            confidence=response.confidence
        )

        col_left, col_right = st.columns([1, 1])

        # Left Column: Conversation Preview
        with col_left:
            st.markdown("#### 💬 Conversation Preview")
            render_customer_message(user_input.strip(), sender_name="Customer")
            
            if response.decision == "ESCALATE_TO_HUMAN":
                render_escalation_notice(
                    reason=response.reason,
                    risk_signals=response.metadata.get("risk_signals", [])
                )
            
            render_agent_message(
                text=response.reply,
                brand_name="AppleSupport AI",
                intent=format_intent_name(response.intent),
                grounded=response.metadata.get("grounded", True)
            )

            # Copy Draft Response Box
            st.markdown("##### 📋 Agent Draft Response")
            st.code(response.reply, language="text")

            # Execution Latency Breakdown
            render_latency_breakdown(response.metadata)

        # Right Column: Deep Explainability & Evidence
        with col_right:
            st.markdown("#### 🔍 Explainability & Safety Inspector")
            
            exp_tab1, exp_tab2, exp_tab3 = st.tabs(["🏷️ Intent & Margin", "⚠️ Risk Signals", "📚 Retrieved Evidence"])

            with exp_tab1:
                st.markdown(f"**Predicted Intent:** `{response.intent}`")
                st.markdown(f"**Confidence Score:** `{format_percentage(response.confidence)}`")
                
                margin = response.metadata.get("confidence_margin", 0.0)
                st.markdown(f"**Confidence Margin (Top-1 vs Top-2):** `{margin:.4f}`")
                
                sim_score = response.metadata.get("semantic_similarity", 0.0)
                if sim_score:
                    st.markdown(f"**Semantic Prototype Cosine:** `{sim_score:.4f}`")

                kw_ev = response.metadata.get("keyword_evidence", [])
                if kw_ev:
                    st.markdown(f"**Detected Keywords:** {', '.join([f'`{k}`' for k in kw_ev])}")

                # Display alternative candidate scores if available
                scores_dict = {response.intent: response.confidence}
                render_alternative_intents_chart(scores_dict, response.intent)

            with exp_tab2:
                risk_signals = response.metadata.get("risk_signals", [])
                render_risk_signals(risk_signals)

                safety_notes = response.metadata.get("safety_notes", [])
                if safety_notes:
                    st.markdown("##### 🛡️ Guardrails Enforced:")
                    for note in safety_notes:
                        st.markdown(f"- {note}")

                is_grounded = response.metadata.get("grounded", True)
                if is_grounded:
                    st.success("✅ Output response is strictly grounded in verified historical support knowledge.")
                else:
                    st.warning("⚠️ Low grounding confidence: generic escalation template applied.")

            with exp_tab3:
                st.markdown(f"**Top Retrieved Support Cases ({len(response.evidence)}):**")
                render_retrieved_evidence(response.evidence)

    elif should_run and not user_input.strip():
        st.warning("Please enter a customer message or select a preset scenario.")
