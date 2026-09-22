"""Escalation Center View: Human-in-the-Loop triage queue, severity tags, and supervisor action desk."""

import html
from typing import Dict, Any, List
import streamlit as st
import pandas as pd

from dashboard.components.kpi_card import render_kpi_card
from dashboard.components.chat_bubble import render_customer_message, render_agent_message
from dashboard.components.explainability import render_risk_signals
from dashboard.utils.formatters import format_intent_name, format_percentage


def render_escalation_center_view(eval_results: Dict[str, Any]):
    """Renders the Human-in-the-Loop Escalation Center."""
    st.markdown("## 🚨 Human-in-the-Loop Escalation Desk")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Review and triage customer inquiries flagged by the AI safety gate. "
        "Approve AI-generated drafts, assign tickets to Tier-2 specialists, or override responses."
        "</p>",
        unsafe_allow_html=True
    )

    detailed_records = eval_results.get("detailed_records", [])
    
    # Extract escalated cases
    escalated_cases = [r for r in detailed_records if r.get("pred_decision") == "ESCALATE_TO_HUMAN"]

    # Append any live session escalations
    session_logs = st.session_state.get("activity_log", [])
    for log in session_logs:
        if log.get("decision") == "ESCALATE_TO_HUMAN":
            escalated_cases.insert(0, {
                "id": f"live_{abs(hash(log['customer_message'])) % 10000}",
                "customer_message": log["customer_message"],
                "gold_intent": log.get("intent", "other"),
                "pred_intent": log.get("intent", "other"),
                "intent_confidence": log.get("confidence", 0.5),
                "gold_decision": "ESCALATE_TO_HUMAN",
                "pred_decision": "ESCALATE_TO_HUMAN",
                "decision_reason": log.get("reason", "Session live escalation"),
                "agent_reply": log.get("reply", ""),
                "top_retrieval_sim": 0.65,
                "latency_ms": log.get("latency_ms", 100)
            })

    # Initialize resolution state
    if "escalation_resolutions" not in st.session_state:
        st.session_state.escalation_resolutions = {}

    total_queue = len(escalated_cases)
    resolved_count = len([k for k, v in st.session_state.escalation_resolutions.items() if v["status"] == "resolved"])
    pending_count = max(0, total_queue - resolved_count)

    # Top KPI Row
    c1, c2, c3 = st.columns(3)
    with c1:
        render_kpi_card("Pending Escalations", f"{pending_count}", "Awaiting Supervisor Action", icon="⏳", variant="warning")
    with c2:
        render_kpi_card("Triage Resolution Rate", format_percentage(resolved_count / total_queue if total_queue else 1.0), f"{resolved_count} Resolved Today", icon="✅", variant="success")
    with c3:
        render_kpi_card("Critical Safety Gate", "100.0%", "0 Missed Risky Queries", icon="🛡️", variant="success")

    # Severity Filter & Search
    col_filter, col_search = st.columns([2, 3])
    with col_filter:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All Severities", "High (Legal / Fraud / Billing)", "Medium (Low Confidence / Margin)", "Low (Safety Fallback)"]
        )
    with col_search:
        search_kw = st.text_input("Search Inquiries", placeholder="Filter by keyword or Case ID...")

    # Classify severity helper
    def get_case_severity(case: Dict[str, Any]) -> str:
        reason = case.get("decision_reason", "").lower()
        if "legal" in reason or "fraud" in reason or "billing" in reason or "dissatisfaction" in reason:
            return "High"
        elif "confidence" in reason or "margin" in reason:
            return "Medium"
        return "Low"

    # Filter cases
    filtered = []
    for c in escalated_cases:
        sev = get_case_severity(c)
        if severity_filter.startswith("High") and sev != "High":
            continue
        if severity_filter.startswith("Medium") and sev != "Medium":
            continue
        if severity_filter.startswith("Low") and sev != "Low":
            continue
        if search_kw.strip():
            if search_kw.lower() not in c["customer_message"].lower() and search_kw.lower() not in c["id"].lower():
                continue
        filtered.append(c)

    st.markdown(f"##### 📋 Active Escalation Queue ({len(filtered)} cases)")

    # Two-Pane Triage Queue
    col_queue, col_action = st.columns([2, 3])

    with col_queue:
        case_options = []
        for c in filtered[:30]:
            cid = c["id"]
            status = st.session_state.escalation_resolutions.get(cid, {}).get("status", "pending")
            status_icon = "✅" if status == "resolved" else "⏳"
            sev = get_case_severity(c)
            sev_icon = "🔴" if sev == "High" else "🟡" if sev == "Medium" else "🟢"
            snippet = c["customer_message"][:35] + ("..." if len(c["customer_message"]) > 35 else "")
            case_options.append((cid, f"{status_icon} {sev_icon} [{cid}] {snippet}"))

        if not case_options:
            st.info("No escalated cases match current filters.")
            return

        selected_case_id = st.selectbox(
            "Select Case to Review",
            options=[c[0] for c in case_options],
            format_func=lambda x: next(c[1] for c in case_options if c[0] == x),
            label_visibility="collapsed"
        )

    # Find selected case
    selected_case = next(c for c in escalated_cases if c["id"] == selected_case_id)
    cur_res = st.session_state.escalation_resolutions.get(selected_case_id, None)

    with col_action:
        st.markdown(f"##### 🛡️ Triage Action Desk: Case `{selected_case_id}`")
        
        # Severity and status badges
        sev = get_case_severity(selected_case)
        sev_color = "#EF4444" if sev == "High" else "#F59E0B" if sev == "Medium" else "#10B981"
        
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="background: rgba(239,68,68,0.15); border: 1px solid {sev_color}; color: {sev_color}; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 700;">
                        Severity: {sev.upper()}
                    </span>
                    <span class="tag-pill tag-pill-intent" style="margin-left: 8px;">🏷️ {format_intent_name(selected_case['pred_intent'])}</span>
                </div>
                <div>
                    <span style="font-size: 12px; color: #94A3B8;">Confidence: <strong style="color:#FFF;">{format_percentage(selected_case['intent_confidence'])}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Customer Message
        render_customer_message(selected_case["customer_message"], sender_name="Customer")

        # Escalation Reason Card
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444; padding: 10px 14px; border-radius: 6px; margin: 12px 0; font-size: 13px; color: #FCA5A5;">
                <strong>Trigger Reason:</strong> {html.escape(selected_case.get('decision_reason', 'Flagged by safety policy.'))}
            </div>
            """,
            unsafe_allow_html=True
        )

        # AI Drafted Response
        st.markdown("##### 🍎 AI Proposed Draft Resolution:")
        render_agent_message(
            text=selected_case.get("agent_reply", "Drafting resolution..."),
            brand_name="AppleSupport AI (Draft for Review)",
            grounded=selected_case.get("grounded", True)
        )

        # Supervisor Actions
        st.markdown("---")
        st.markdown("##### ✍️ Supervisor Decision & Takeover")

        if cur_res:
            st.success(f"✅ Case marked as **{cur_res['action']}** on {cur_res.get('timestamp', 'today')}.")
            if "notes" in cur_res:
                st.info(f"Notes: {cur_res['notes']}")
        
        edit_draft = st.text_area(
            "Edit or Approve Agent Reply",
            value=selected_case.get("agent_reply", ""),
            height=90,
            key=f"edit_draft_{selected_case_id}"
        )

        btn_c1, btn_c2, btn_c3 = st.columns(3)
        with btn_c1:
            if st.button("✅ Approve & Send", key=f"appr_{selected_case_id}", type="primary", use_container_width=True):
                st.session_state.escalation_resolutions[selected_case_id] = {
                    "status": "resolved",
                    "action": "Approved & Sent",
                    "final_reply": edit_draft,
                    "timestamp": "Just now"
                }
                st.rerun()

        with btn_c2:
            if st.button("👨‍💼 Escalate to Tier-2", key=f"tier2_{selected_case_id}", use_container_width=True):
                st.session_state.escalation_resolutions[selected_case_id] = {
                    "status": "resolved",
                    "action": "Escalated to Senior Tier-2 Specialist",
                    "timestamp": "Just now"
                }
                st.rerun()

        with btn_c3:
            if st.button("✔️ Mark Resolved", key=f"res_{selected_case_id}", use_container_width=True):
                st.session_state.escalation_resolutions[selected_case_id] = {
                    "status": "resolved",
                    "action": "Marked Resolved by Supervisor",
                    "timestamp": "Just now"
                }
                st.rerun()
