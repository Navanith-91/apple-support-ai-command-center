"""Chat bubble component for customer and support agent messages."""

import html
from typing import Optional, List
import streamlit as st


def render_customer_message(text: str, sender_name: str = "Customer", timestamp: Optional[str] = None):
    """Renders a customer chat bubble."""
    safe_text = html.escape(text)
    ts_str = f'<span style="float:right; font-weight:normal; color:#64748B;">{timestamp}</span>' if timestamp else ""
    bubble_html = f"""
    <div class="chat-bubble-customer">
        <div class="bubble-sender customer">👤 {sender_name} {ts_str}</div>
        <p class="bubble-text">{safe_text}</p>
    </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)


def render_agent_message(
    text: str,
    brand_name: str = "AppleSupport AI",
    intent: Optional[str] = None,
    grounded: bool = True,
    timestamp: Optional[str] = None
):
    """Renders an AI Agent resolution reply bubble."""
    safe_text = html.escape(text)
    ts_str = f'<span style="float:right; font-weight:normal; color:#64748B;">{timestamp}</span>' if timestamp else ""
    grounding_badge = ' <span class="tag-pill" style="color:#34D399; border-color:rgba(16,185,129,0.4);">🛡️ Grounded</span>' if grounded else ""
    intent_badge = f' <span class="tag-pill tag-pill-intent">🏷️ {intent}</span>' if intent else ""
    
    bubble_html = f"""
    <div class="chat-bubble-agent">
        <div class="bubble-sender agent">🍎 {brand_name} {grounding_badge}{intent_badge} {ts_str}</div>
        <p class="bubble-text">{safe_text}</p>
    </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)


def render_escalation_notice(reason: str, risk_signals: Optional[List[str]] = None):
    """Renders an escalation notice banner inside conversation."""
    safe_reason = html.escape(reason)
    signals_html = ""
    if risk_signals:
        signals_html = "".join([f'<span class="tag-pill tag-pill-risk">⚠️ {html.escape(s)}</span>' for s in risk_signals])
    
    notice_html = f"""
    <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px 16px; margin: 12px 0;">
        <div style="color: #F87171; font-weight: 700; font-size: 13px; margin-bottom: 4px;">
            🚨 ESCALATED TO HUMAN SUPPORT SPECIALIST
        </div>
        <div style="color: #FCA5A5; font-size: 12px; margin-bottom: 6px;">{safe_reason}</div>
        <div>{signals_html}</div>
    </div>
    """
    st.markdown(notice_html, unsafe_allow_html=True)
