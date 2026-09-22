"""Top Command Center Header component with live status badges."""

import streamlit as st


def render_header(brand_name: str = "AppleSupport", model_name: str = "Hybrid TF-IDF + MiniLM"):
    """Renders the persistent top header banner."""
    header_html = f"""
    <div class="command-center-header">
        <div class="header-title-box">
            <div class="header-logo-badge">🍎</div>
            <div>
                <h1 class="header-title">{brand_name} AI Command Center</h1>
                <p class="header-subtitle">Enterprise Autonomous Support Agent • Safety-First Escalation Engine</p>
            </div>
        </div>
        <div class="header-badges">
            <div class="badge-live">SYSTEM ONLINE</div>
            <div class="badge-model">⚙️ {model_name}</div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
