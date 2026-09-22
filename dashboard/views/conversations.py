"""Conversations Inbox View: Real multi-turn ticket browser with live AI agent auditing."""

import html
from typing import Dict, Any
import streamlit as st
import pandas as pd

from src.agent import CustomerSupportAgent
from dashboard.components.chat_bubble import render_customer_message, render_agent_message, render_escalation_notice
from dashboard.components.explainability import render_decision_badge, render_retrieved_evidence
from dashboard.utils.formatters import format_intent_name, format_percentage


def render_conversations_view(conversations_df: pd.DataFrame, agent: CustomerSupportAgent):
    """Renders the Support Conversations Inbox with ticket browser and agent auditor."""
    st.markdown("## 📥 Customer Support Inbox")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Explore real historical AppleSupport customer conversation threads. "
        "Inspect original human agent resolutions and run real-time AI agent audits on any ticket."
        "</p>",
        unsafe_allow_html=True
    )

    if conversations_df.empty:
        st.error("No conversation archive data available.")
        return

    # Top Filters Row
    col_search, col_filter, col_count = st.columns([3, 2, 2])
    
    with col_search:
        search_query = st.text_input("🔍 Search Inbound Text", placeholder="e.g., battery, screen, update, refund...")
    
    # Filter dataset
    filtered_df = conversations_df.copy()
    if search_query.strip():
        mask = filtered_df["customer_text_clean"].str.contains(search_query.strip(), case=False, na=False) | \
               filtered_df["agent_text_clean"].str.contains(search_query.strip(), case=False, na=False)
        filtered_df = filtered_df[mask]

    with col_count:
        st.markdown(
            f"<div style='padding-top: 32px; color: #94A3B8; font-size: 13px;'>"
            f"Showing <strong style='color:#FFFFFF;'>{len(filtered_df):,}</strong> tickets"
            f"</div>",
            unsafe_allow_html=True
        )

    # Two-Pane Layout: Left ticket list, Right detail viewer
    col_list, col_detail = st.columns([2, 3])

    with col_list:
        st.markdown("##### 📋 Ticket Queue")
        
        # Display scrollable ticket list
        ticket_options = []
        for idx, row in filtered_df.head(25).iterrows():
            conv_id = str(row["conversation_id"])[:10]
            snippet = str(row["customer_text_clean"])[:45] + ("..." if len(str(row["customer_text_clean"])) > 45 else "")
            ticket_options.append((idx, f"#{conv_id} - {snippet}"))

        if not ticket_options:
            st.info("No tickets match the search query.")
            return

        selected_ticket_idx = st.selectbox(
            "Select Ticket",
            options=[t[0] for t in ticket_options],
            format_func=lambda x: next(t[1] for t in ticket_options if t[0] == x),
            label_visibility="collapsed"
        )

    # Right Canvas: Selected Ticket Details & AI Audit
    selected_row = filtered_df.loc[selected_ticket_idx]
    
    with col_detail:
        st.markdown("##### 📄 Conversation Detail")
        
        ticket_id = selected_row["conversation_id"]
        customer_msg = selected_row["customer_text_clean"]
        human_agent_reply = selected_row["agent_text_clean"]

        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: #38BDF8; font-weight: 700; font-size: 13px;">Ticket ID: {ticket_id}</span>
                    <span class="tag-pill">Brand: AppleSupport</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Render original customer inquiry & historical human reply
        render_customer_message(customer_msg, sender_name="Customer")
        render_agent_message(human_agent_reply, brand_name="AppleSupport (Human Specialist)", grounded=True)

        st.markdown("---")
        st.markdown("##### 🤖 Real-Time AI Agent Audit")
        
        # Live audit execution on this ticket
        if st.button("⚡ Run AI Agent Analysis on this Ticket", type="primary", key="audit_btn"):
            with st.spinner("Analyzing message through AI pipeline..."):
                audit_response = agent.run(customer_msg)
            
            render_decision_badge(
                decision=audit_response.decision,
                reason=audit_response.reason,
                confidence=audit_response.confidence
            )

            st.markdown(f"**AI Predicted Intent:** `{audit_response.intent}` ({format_percentage(audit_response.confidence)})")
            
            if audit_response.decision == "ESCALATE_TO_HUMAN":
                render_escalation_notice(
                    reason=audit_response.reason,
                    risk_signals=audit_response.metadata.get("risk_signals", [])
                )
            
            st.markdown("##### 🍎 AI Proposed Resolution:")
            render_agent_message(
                text=audit_response.reply,
                brand_name="AppleSupport AI",
                intent=format_intent_name(audit_response.intent),
                grounded=audit_response.metadata.get("grounded", True)
            )

            with st.expander("🔍 View Grounding Evidence Used"):
                render_retrieved_evidence(audit_response.evidence)
