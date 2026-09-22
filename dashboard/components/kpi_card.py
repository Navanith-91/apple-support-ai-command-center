"""KPI metric card component with status styling and delta badges."""

import streamlit as st


def render_kpi_card(
    label: str,
    value: str,
    subtext: str,
    icon: str = "📊",
    variant: str = "default",
    delta_type: str = "neutral"
):
    """Renders a custom HTML KPI metric card.
    
    Args:
        label: Metric title (e.g. 'Automation Rate')
        value: Main metric value string (e.g. '94.2%')
        subtext: Context string or delta (e.g. '+3.4% vs baseline')
        icon: Emoji or icon symbol
        variant: 'default', 'success', 'warning', or 'danger'
        delta_type: 'positive', 'negative', or 'neutral'
    """
    card_html = f"""
    <div class="kpi-container {variant}">
        <div class="kpi-header">
            <span class="kpi-label">{label}</span>
            <span class="kpi-icon">{icon}</span>
        </div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtext">
            <span class="kpi-delta-{delta_type}">{subtext}</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
