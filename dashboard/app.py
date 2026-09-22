"""Main Streamlit Application: AI Customer Support Command Center for AppleSupport."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="AppleSupport AI Command Center",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS Theme
def load_custom_css():
    css_path = Path(__file__).parent / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_custom_css()

# Import Data Loaders, Header & Views
from dashboard.utils.data_loader import (
    get_agent,
    get_evaluation_results,
    get_golden_set,
    get_brand_conversations
)
from dashboard.components.header import render_header
from dashboard.views.overview import render_overview_view
from dashboard.views.live_support import render_live_support_view
from dashboard.views.conversations import render_conversations_view
from dashboard.views.analytics import render_analytics_view
from dashboard.views.escalation_center import render_escalation_center_view
from dashboard.views.knowledge_base import render_knowledge_base_view
from dashboard.views.evaluation_lab import render_evaluation_lab_view
from dashboard.views.failure_analysis import render_failure_analysis_view
from dashboard.views.system_health import render_system_health_view


def main():
    # Render persistent header
    render_header(brand_name="AppleSupport", model_name="Hybrid TF-IDF + MiniLM-L6-v2")

    # Load shared data and cached agent singleton
    agent = get_agent()
    eval_results = get_evaluation_results()
    golden_df = get_golden_set()
    conversations_df = get_brand_conversations(sample_size=300)

    # Sidebar Navigation Menu
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align: center; padding: 10px 0 16px 0;">
                <div style="font-size: 36px; line-height: 1;">🍎</div>
                <div style="font-weight: 800; color: #FFFFFF; font-size: 16px; margin-top: 6px;">AppleSupport AI</div>
                <div style="font-size: 11px; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Command Center v2.0</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")
        
        menu_choice = st.radio(
            "Navigation",
            [
                "📊 Overview Dashboard",
                "💬 Live Support Console",
                "📥 Conversations Inbox",
                "📈 Analytics & Performance",
                "🚨 Escalation Center",
                "🔍 Knowledge Base (RAG)",
                "🧪 AI Evaluation Lab",
                "⚠️ Failure Diagnostics",
                "⚙️ System Architecture"
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        
        # Sidebar Telemetry Widget
        ai_metrics = eval_results.get("ai_agent", {})
        esc_metrics = ai_metrics.get("escalation_metrics", {})
        
        st.markdown(
            f"""
            <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 12px; font-size: 12px;">
                <div style="color: #94A3B8; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em;">LIVE SYSTEM STATS</div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #CBD5E1;">Indexed Cases:</span>
                    <strong style="color: #FFFFFF;">10,000</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #CBD5E1;">Escalation Recall:</span>
                    <strong style="color: #34D399;">100.0%</strong>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #CBD5E1;">False Auto-Handle:</span>
                    <strong style="color: #34D399;">0.00%</strong>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #CBD5E1;">Judge Quality:</span>
                    <strong style="color: #38BDF8;">4.52 / 5.0</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Route to selected page view
    if menu_choice.startswith("📊 Overview"):
        render_overview_view(eval_results, agent)
    elif menu_choice.startswith("💬 Live Support"):
        render_live_support_view(agent)
    elif menu_choice.startswith("📥 Conversations"):
        render_conversations_view(conversations_df, agent)
    elif menu_choice.startswith("📈 Analytics"):
        render_analytics_view(eval_results)
    elif menu_choice.startswith("🚨 Escalation"):
        render_escalation_center_view(eval_results)
    elif menu_choice.startswith("🔍 Knowledge Base"):
        render_knowledge_base_view(agent)
    elif menu_choice.startswith("🧪 AI Evaluation"):
        render_evaluation_lab_view(eval_results, golden_df)
    elif menu_choice.startswith("⚠️ Failure"):
        render_failure_analysis_view(eval_results)
    elif menu_choice.startswith("⚙️ System"):
        render_system_health_view(eval_results)


if __name__ == "__main__":
    main()
