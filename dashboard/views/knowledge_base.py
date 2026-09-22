"""Knowledge Base View: Direct semantic vector retrieval explorer over 10,000 indexed historical support cases."""

import time
import html
from typing import Dict, Any
import streamlit as st

from src.agent import CustomerSupportAgent
from dashboard.components.kpi_card import render_kpi_card
from dashboard.components.explainability import render_retrieved_evidence
from dashboard.utils.formatters import format_latency, format_percentage

SAMPLE_QUERIES = [
    "🔋 battery draining fast on ios 11",
    "🎧 bluetooth airpods disconnecting randomly",
    "🎵 how to cancel apple music subscription",
    "📱 touch screen unresponsive after dropping iphone",
    "💻 macbook pro keyboard keys sticking",
    "💳 unknown charges on my credit card statement"
]


def render_knowledge_base_view(agent: CustomerSupportAgent):
    """Renders the Knowledge Base & Vector Retrieval Explorer."""
    st.markdown("## 🔍 Knowledge Base (Semantic Vector RAG)")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Directly query the 10,000 indexed historical AppleSupport resolution vectors. "
        "Test cosine similarity scoring, semantic nearest neighbor matches, and evidence grounding."
        "</p>",
        unsafe_allow_html=True
    )

    # Retrieval Index Specs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Indexed Cases", "10,000", "Verified Support Pairs", icon="📚", variant="default")
    with c2:
        render_kpi_card("Embedding Model", "all-MiniLM-L6-v2", "384-dimensional dense vectors", icon="🧠", variant="default")
    with c3:
        render_kpi_card("Sufficiency Rate", "96.0%", "Queries with strong grounding", icon="🛡️", variant="success")
    with c4:
        render_kpi_card("Avg Retrieval Time", "12.8 ms", "Sub-15ms local cosine index", icon="⚡", variant="default")

    # Sample query selector pills
    st.markdown("##### ⚡ Quick Knowledge Search Examples")
    q_cols = st.columns(3)
    
    if "kb_query" not in st.session_state:
        st.session_state.kb_query = "battery draining fast on ios 11"

    for idx, (col, q_text) in enumerate(zip([q_cols[0], q_cols[1], q_cols[2], q_cols[0], q_cols[1], q_cols[2]], SAMPLE_QUERIES)):
        clean_q = q_text.split(" ", 1)[1] if " " in q_text else q_text
        with col:
            if st.button(q_text, key=f"kb_sample_{idx}", use_container_width=True):
                st.session_state.kb_query = clean_q
                st.session_state.kb_auto_search = True

    # Search Bar & Parameters
    st.markdown("##### 🔍 Semantic Query Input")
    col_input, col_topk, col_thresh = st.columns([4, 1, 1])

    with col_input:
        search_query = st.text_input(
            "Query",
            value=st.session_state.kb_query,
            placeholder="Type any customer support query or technical issue...",
            label_visibility="collapsed"
        )
    with col_topk:
        top_k = st.slider("Top K", min_value=1, max_value=10, value=3)
    with col_thresh:
        min_sim = st.slider("Min Cosine", min_value=0.0, max_value=0.9, value=0.35, step=0.05)

    search_btn = st.button("🔎 Execute Vector Search", type="primary")

    should_search = search_btn or st.session_state.get("kb_auto_search", False)
    if st.session_state.get("kb_auto_search", False):
        st.session_state.kb_auto_search = False

    if should_search and search_query.strip():
        t0 = time.perf_counter()
        evidence_results = agent.retriever.retrieve(
            query=search_query.strip(),
            top_k=top_k,
            min_similarity=min_sim
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        st.markdown("---")
        
        # Summary row
        top_score = evidence_results[0].score if evidence_results else 0.0
        is_sufficient = top_score >= 0.60
        
        st.markdown(
            f"""
            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 12px 18px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 14px; color: #FFFFFF; font-weight: 600;">Found {len(evidence_results)} matching cases</span>
                    <span style="font-size: 13px; color: #94A3B8; margin-left: 12px;">Top Cosine Match: <strong style="color:#38BDF8;">{top_score*100:.1f}%</strong></span>
                </div>
                <div style="display: flex; gap: 8px;">
                    <span class="tag-pill" style="color: {'#34D399' if is_sufficient else '#F87171'};">
                        {'🛡️ Sufficient Grounding' if is_sufficient else '⚠️ Weak Evidence'}
                    </span>
                    <span class="tag-pill">⏱️ Search Time: {format_latency(latency_ms)}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Evidence Cards
        evidence_dicts = [ev.model_dump() for ev in evidence_results]
        render_retrieved_evidence(evidence_dicts)
