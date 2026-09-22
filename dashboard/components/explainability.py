"""Explainability components for inspecting AI agent decisions, confidence, and grounding."""

import html
from typing import List, Dict, Any, Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.utils.formatters import format_intent_name, format_percentage, format_latency


def render_decision_badge(decision: str, reason: str, confidence: float):
    """Renders the top decision card (AUTO_HANDLE vs ESCALATE_TO_HUMAN)."""
    is_auto = decision.upper() == "AUTO_HANDLE"
    badge_class = "decision-badge-auto" if is_auto else "decision-badge-escalate"
    icon = "✅" if is_auto else "🚨"
    title = "AUTOMATED HANDLING APPROVED" if is_auto else "ESCALATED TO HUMAN SPECIALIST"
    
    badge_html = f"""
    <div class="custom-panel" style="border-left: 4px solid {'#10B981' if is_auto else '#EF4444'};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <div class="{badge_class}">
                <span>{icon}</span>
                <span>{title}</span>
            </div>
            <div style="font-size: 13px; color: #94A3B8;">
                Confidence: <strong style="color: #FFFFFF;">{format_percentage(confidence)}</strong>
            </div>
        </div>
        <div style="font-size: 13px; color: #CBD5E1; line-height: 1.5;">
            <strong>Policy Rationale:</strong> {html.escape(reason)}
        </div>
    </div>
    """
    st.markdown(badge_html, unsafe_allow_html=True)


def render_risk_signals(signals: List[str]):
    """Renders list of detected safety and escalation risk triggers."""
    if not signals:
        st.markdown(
            '<div style="color: #10B981; font-size: 13px; padding: 6px 0;">🛡️ <strong>No risk triggers detected.</strong> Safe for automated resolution.</div>',
            unsafe_allow_html=True
        )
        return
    
    badges = "".join([f'<span class="tag-pill tag-pill-risk">⚠️ {html.escape(s)}</span>' for s in signals])
    html_block = f"""
    <div style="margin: 8px 0;">
        <div style="font-size: 12px; font-weight: 600; color: #F87171; margin-bottom: 6px;">SAFETY & ESCALATION RISK SIGNALS DETECTED ({len(signals)}):</div>
        <div>{badges}</div>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)


def render_alternative_intents_chart(scores: Dict[str, float], top_intent: str):
    """Renders a horizontal probability bar chart for candidate intents."""
    if not scores:
        return
    
    # Sort and take top 5
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    df = pd.DataFrame(sorted_scores, columns=["Intent", "Probability"])
    df["IntentLabel"] = df["Intent"].apply(format_intent_name)
    df["IsTop"] = df["Intent"] == top_intent
    
    fig = go.Figure()
    colors = ['#38BDF8' if is_top else '#334155' for is_top in df["IsTop"]]
    
    fig.add_trace(go.Bar(
        x=df["Probability"],
        y=df["IntentLabel"],
        orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{p*100:.1f}%" for p in df["Probability"]],
        textposition='outside',
        textfont=dict(color='#F8FAFC', size=11),
        hoverinfo='x+y'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=40, t=10, b=10),
        height=180,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', range=[0, 1.15], showticklabels=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color='#CBD5E1'))
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_retrieved_evidence(evidence: List[Dict[str, Any]]):
    """Renders the top retrieved historical support cases with similarity scores."""
    if not evidence:
        st.markdown('<div style="color:#94A3B8; font-size:13px;">No historical support evidence met the similarity threshold.</div>', unsafe_allow_html=True)
        return
    
    for i, ev in enumerate(evidence, 1):
        score = ev.get("score", 0.0)
        query = ev.get("customer_query", "")
        reply = ev.get("agent_reply", "")
        conv_id = ev.get("conversation_id", f"case_{i}")
        
        score_color = "#34D399" if score >= 0.70 else "#FBBF24" if score >= 0.50 else "#F87171"
        
        card_html = f"""
        <div class="evidence-card">
            <div class="evidence-header">
                <span>Case #{i} • ID: {conv_id[:12]}</span>
                <span style="color: {score_color}; font-weight: 700;">Cosine Match: {score*100:.1f}%</span>
            </div>
            <div class="evidence-query">
                <span style="color:#64748B; font-weight:600;">Customer Inbound:</span> "{html.escape(query)}"
            </div>
            <div class="evidence-reply">
                <span style="color:#38BDF8; font-weight:600;">Historical Resolution:</span> "{html.escape(reply)}"
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)


def render_latency_breakdown(metadata: Dict[str, Any]):
    """Renders execution latency pills across pipeline stages."""
    total = metadata.get("total_latency_ms", 0.0)
    clf = metadata.get("classification_latency_ms", 0.0)
    ret = metadata.get("retrieval_latency_ms", 0.0)
    gen = metadata.get("generation_latency_ms", 0.0)
    
    html_block = f"""
    <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px;">
        <span class="tag-pill" style="color: #38BDF8; border-color: rgba(56, 189, 248, 0.3);">⏱️ Total: {format_latency(total)}</span>
        <span class="tag-pill">🏷️ Intent: {format_latency(clf)}</span>
        <span class="tag-pill">🔍 Vector RAG: {format_latency(ret)}</span>
        <span class="tag-pill">✍️ Grounded Gen: {format_latency(gen)}</span>
    </div>
    """
    st.markdown(html_block, unsafe_allow_html=True)
