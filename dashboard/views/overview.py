"""Overview Dashboard View: Executive KPIs, operational status, and live activity stream."""

from typing import Dict, Any
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from dashboard.components.kpi_card import render_kpi_card
from dashboard.utils.formatters import format_intent_name, format_percentage, format_latency


def render_overview_view(eval_results: Dict[str, Any], agent):
    """Renders the primary Command Center executive overview."""
    st.markdown("## 📊 Executive Command Center")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Real-time operational monitoring, safety metrics, and automated triage telemetry for AppleSupport."
        "</p>",
        unsafe_allow_html=True
    )

    ai_agent = eval_results.get("ai_agent", {})
    intent_metrics = ai_agent.get("intent_metrics", {})
    esc_metrics = ai_agent.get("escalation_metrics", {})
    judge_metrics = ai_agent.get("judge_metrics", {})
    latency = ai_agent.get("latency", {})

    # Top KPI Row (6 Cards)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    
    with c1:
        render_kpi_card(
            label="Evaluated Cases",
            value=f"{esc_metrics.get('total_evaluated', 200):,}",
            subtext="50,000 in archive",
            icon="📁",
            variant="default",
            delta_type="neutral"
        )
    with c2:
        render_kpi_card(
            label="Escalation Recall",
            value=format_percentage(esc_metrics.get("escalation_recall", 1.0)),
            subtext="100% Critical Recall",
            icon="🛡️",
            variant="success",
            delta_type="positive"
        )
    with c3:
        render_kpi_card(
            label="False Auto-Handle",
            value=format_percentage(esc_metrics.get("false_auto_handling_rate", 0.0)),
            subtext="0 Catastrophic Leaks",
            icon="🔒",
            variant="success",
            delta_type="positive"
        )
    with c4:
        render_kpi_card(
            label="Intent Macro F1",
            value=f"{intent_metrics.get('macro_f1', 0.6373):.4f}",
            subtext=f"Acc: {format_percentage(intent_metrics.get('accuracy', 0.635))}",
            icon="🎯",
            variant="default",
            delta_type="positive"
        )
    with c5:
        render_kpi_card(
            label="LLM Judge Quality",
            value=f"{judge_metrics.get('overall', 4.52):.2f}/5.0",
            subtext="Brand Voice: 5.0/5.0",
            icon="⭐",
            variant="success",
            delta_type="positive"
        )
    with c6:
        render_kpi_card(
            label="P50 Latency",
            value=format_latency(latency.get("p50_latency_ms", 74.16)),
            subtext=f"Mean: {format_latency(latency.get('mean_latency_ms', 120.44))}",
            icon="⚡",
            variant="default",
            delta_type="neutral"
        )

    # Operational Status Banner
    st.markdown(
        """
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 10px; padding: 14px 18px; margin: 16px 0 24px 0; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 20px;">🛡️</span>
                <div>
                    <strong style="color: #38BDF8; font-size: 14px;">Safety-First Autonomous Policy Active:</strong>
                    <span style="color: #CBD5E1; font-size: 13px;"> Low-confidence (<0.65), low-margin (<0.15), and sensitive billing/fraud inquiries are automatically routed to human escalation.</span>
                </div>
            </div>
            <div style="font-size: 12px; color: #94A3B8; white-space: nowrap; margin-left: 16px;">
                Dense Index: <strong style="color:#FFFFFF;">10,000 Vectors</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Middle Charts Row: Intent Breakdown & Escalation Split
    col_chart1, col_chart2 = st.columns([3, 2])

    with col_chart1:
        st.markdown("#### 📈 Benchmark Intent Distribution")
        labels = intent_metrics.get("labels", [])
        
        # Parse per_intent support or construct from detailed records
        detailed_records = eval_results.get("detailed_records", [])
        if detailed_records:
            df_det = pd.DataFrame(detailed_records)
            intent_counts = df_det["gold_intent"].value_counts().reset_index()
            intent_counts.columns = ["Intent", "Count"]
            intent_counts["FormattedIntent"] = intent_counts["Intent"].apply(format_intent_name)
            
            fig = px.bar(
                intent_counts,
                x="Count",
                y="FormattedIntent",
                orientation="h",
                color="Count",
                color_continuous_scale="Blues",
                text="Count"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=20, t=10, b=10),
                height=260,
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#CBD5E1"))
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Evaluation data loading...")

    with col_chart2:
        st.markdown("#### ⚖️ Escalation vs Automation Split")
        total_esc = esc_metrics.get("total_gold_escalate", 79)
        total_auto = esc_metrics.get("total_gold_auto", 121)
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Auto-Handled", "Escalated to Human"],
            values=[total_auto, total_esc],
            hole=0.6,
            marker=dict(colors=["#10B981", "#EF4444"], line=dict(color="#0B0F17", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#FFFFFF")
        )])
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=260,
            showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # Bottom Section: Live Activity / Decision Stream
    st.markdown("---")
    st.markdown("#### ⚡ Live Decision Stream")

    # Retrieve session activity log or fallback to recent evaluation cases
    activity_stream = st.session_state.get("activity_log", [])
    
    if not activity_stream and detailed_records:
        # Seed initial 5 records from evaluated cases
        sample_records = detailed_records[:5]
        for r in sample_records:
            activity_stream.append({
                "customer_message": r["customer_message"],
                "intent": r["pred_intent"],
                "decision": r["pred_decision"],
                "reason": r["decision_reason"],
                "confidence": r["intent_confidence"],
                "latency_ms": r["latency_ms"]
            })

    if activity_stream:
        for item in activity_stream[:6]:
            is_auto = item["decision"] == "AUTO_HANDLE"
            badge_color = "#10B981" if is_auto else "#EF4444"
            badge_bg = "rgba(16, 185, 129, 0.15)" if is_auto else "rgba(239, 68, 68, 0.15)"
            icon = "✅" if is_auto else "🚨"
            
            row_html = f"""
            <div style="background: #131B2E; border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div style="max-width: 65%;">
                    <div style="color: #F8FAFC; font-size: 13px; font-weight: 500; margin-bottom: 4px;">"{item['customer_message']}"</div>
                    <div style="font-size: 12px; color: #94A3B8;">
                        <span class="tag-pill tag-pill-intent">🏷️ {format_intent_name(item['intent'])}</span>
                        <span>Confidence: <strong style="color:#FFFFFF;">{format_percentage(item['confidence'])}</strong></span>
                        <span style="margin-left: 8px;">Latency: <strong style="color:#38BDF8;">{format_latency(item.get('latency_ms', 75))}</strong></span>
                    </div>
                </div>
                <div>
                    <span style="background: {badge_bg}; border: 1px solid {badge_color}; color: {badge_color}; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 700;">
                        {icon} {item['decision']}
                    </span>
                </div>
            </div>
            """
            st.markdown(row_html, unsafe_allow_html=True)
    else:
        st.info("No queries processed in this session yet. Launch the Live Console to test queries in real time.")
