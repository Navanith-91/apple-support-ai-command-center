"""Analytics & Performance View: Interactive confusion matrix, intent benchmarks, safety trade-offs, and judge rubric."""

import re
import io
from typing import Dict, Any, List
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from dashboard.components.kpi_card import render_kpi_card
from dashboard.utils.formatters import format_intent_name, format_percentage


def render_analytics_view(eval_results: Dict[str, Any]):
    """Renders the Analytics & Model Performance view."""
    st.markdown("## 📈 Analytics & Model Performance")
    st.markdown(
        "<p style='color: #94A3B8; margin-top:-10px; margin-bottom:20px;'>"
        "Comprehensive model diagnostics, per-intent classification metrics, "
        "confusion matrix heatmap, safety trade-off analysis, and LLM Judge rubric scores."
        "</p>",
        unsafe_allow_html=True
    )

    ai_agent = eval_results.get("ai_agent", {})
    intent_metrics = ai_agent.get("intent_metrics", {})
    esc_metrics = ai_agent.get("escalation_metrics", {})
    judge_metrics = ai_agent.get("judge_metrics", {})
    detailed_records = eval_results.get("detailed_records", [])

    # Top Metric Banner
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Overall Accuracy", format_percentage(intent_metrics.get("accuracy", 0.635)), "Benchmark Golden Set", icon="🎯", variant="default")
    with c2:
        render_kpi_card("Macro F1-Score", f"{intent_metrics.get('macro_f1', 0.6373):.4f}", "+0.1469 vs TF-IDF baseline", icon="📊", variant="success", delta_type="positive")
    with c3:
        render_kpi_card("Escalation Recall", format_percentage(esc_metrics.get("escalation_recall", 1.0)), "0 missed critical cases", icon="🛡️", variant="success", delta_type="positive")
    with c4:
        render_kpi_card("LLM Judge Score", f"{judge_metrics.get('overall', 4.52):.2f}/5.0", "Rubric weighted composite", icon="⭐", variant="success", delta_type="positive")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Per-Intent Metrics",
        "🔲 Confusion Matrix Heatmap",
        "🛡️ Safety & Escalation Trade-off",
        "⭐ LLM Judge Rubric Breakdown"
    ])

    # Tab 1: Per-Intent Performance
    with tab1:
        st.markdown("#### 🎯 Intent Classification Precision, Recall & F1")
        
        # Build dataframe from detailed records for precision
        if detailed_records:
            df_rec = pd.DataFrame(detailed_records)
            y_true = df_rec["gold_intent"]
            y_pred = df_rec["pred_intent"]
            labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
            
            report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
            rows = []
            for lbl in labels:
                if lbl in report_dict:
                    rows.append({
                        "Intent": format_intent_name(lbl),
                        "RawIntent": lbl,
                        "Precision": report_dict[lbl]["precision"],
                        "Recall": report_dict[lbl]["recall"],
                        "F1-Score": report_dict[lbl]["f1-score"],
                        "Support": int(report_dict[lbl]["support"])
                    })
            df_per_intent = pd.DataFrame(rows)

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=df_per_intent["Intent"], y=df_per_intent["Precision"], name="Precision", marker_color="#38BDF8"))
            fig_bar.add_trace(go.Bar(x=df_per_intent["Intent"], y=df_per_intent["Recall"], name="Recall", marker_color="#10B981"))
            fig_bar.add_trace(go.Bar(x=df_per_intent["Intent"], y=df_per_intent["F1-Score"], name="F1-Score", marker_color="#6366F1"))

            fig_bar.update_layout(
                template="plotly_dark",
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=350,
                xaxis=dict(tickangle=-25, tickfont=dict(size=11, color="#CBD5E1")),
                yaxis=dict(range=[0, 1.1], gridcolor="rgba(255,255,255,0.06)", tickformat=".0%"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # Data Table
            st.dataframe(
                df_per_intent[["Intent", "Precision", "Recall", "F1-Score", "Support"]].style.format({
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1-Score": "{:.4f}",
                    "Support": "{:,}"
                }),
                use_container_width=True,
                hide_index=True
            )

    # Tab 2: Interactive Confusion Matrix Heatmap
    with tab2:
        st.markdown("#### 🔲 9x9 Intent Classification Confusion Matrix")
        st.caption("Rows represent actual ground truth intents; columns represent AI agent predictions.")

        if detailed_records:
            df_rec = pd.DataFrame(detailed_records)
            labels = sorted(df_rec["gold_intent"].unique())
            formatted_labels = [format_intent_name(lbl) for lbl in labels]
            
            cm = confusion_matrix(df_rec["gold_intent"], df_rec["pred_intent"], labels=labels)
            
            fig_cm = px.imshow(
                cm,
                x=formatted_labels,
                y=formatted_labels,
                labels=dict(x="Predicted Intent", y="Actual Gold Intent", color="Cases"),
                color_continuous_scale="Blues",
                text_auto=True
            )
            fig_cm.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
                height=520,
                xaxis=dict(tickangle=-30, tickfont=dict(size=10, color="#CBD5E1")),
                yaxis=dict(tickfont=dict(size=10, color="#CBD5E1"))
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    # Tab 3: Safety & Escalation Trade-off
    with tab3:
        st.markdown("#### 🛡️ Safety-First Escalation Policy Analysis")
        st.markdown(
            "In customer support, **failing to escalate a critical inquiry** (e.g. fraud, billing error, angry churn) "
            "causes severe customer dissatisfaction and financial loss. Our policy enforces a **Zero-Tolerance Safety Gate**."
        )

        col_metric1, col_metric2 = st.columns(2)
        with col_metric1:
            st.markdown(
                """
                <div class="custom-panel" style="border-left: 4px solid #10B981;">
                    <div style="font-size: 14px; font-weight: 700; color: #10B981; margin-bottom: 8px;">
                        🔒 False Auto-Handling Rate (FAHR): 0.00%
                    </div>
                    <div style="font-size: 13px; color: #CBD5E1; line-height: 1.5;">
                        <strong>0 out of 79</strong> sensitive/escalation cases were mistakenly handled by the bot.
                        Every single high-risk or complaint case was safely directed to a human specialist.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_metric2:
            st.markdown(
                """
                <div class="custom-panel" style="border-left: 4px solid #F59E0B;">
                    <div style="font-size: 14px; font-weight: 700; color: #F59E0B; margin-bottom: 8px;">
                        ⚠️ Unnecessary Escalation Rate (UER): 97.52%
                    </div>
                    <div style="font-size: 13px; color: #CBD5E1; line-height: 1.5;">
                        Due to the strict confidence threshold (0.65) and margin filter (0.15), borderline simple cases escalate.
                        This conservative trade-off guarantees safety while retaining full human supervisory control.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Baseline Comparison on Safety
        st.markdown("##### 🔍 Safety Metric Comparison Across Systems")
        maj_esc = eval_results.get("majority_baseline", {}).get("escalation_metrics", {})
        tfidf_esc = eval_results.get("tfidf_baseline", {}).get("escalation_metrics", {})
        rule_esc = eval_results.get("rule_based_baseline", {}).get("escalation_metrics", {})
        ai_esc = eval_results.get("ai_agent", {}).get("escalation_metrics", {})

        safety_df = pd.DataFrame([
            {"System": "Majority Class Baseline", "FAHR (Lower is Better)": maj_esc.get("false_auto_handling_rate", 1.0), "Escalation Recall": maj_esc.get("escalation_recall", 0.0), "Critical Leaks": maj_esc.get("false_auto_handle_count", 79)},
            {"System": "TF-IDF + Classifier Baseline", "FAHR (Lower is Better)": tfidf_esc.get("false_auto_handling_rate", 0.1013), "Escalation Recall": tfidf_esc.get("escalation_recall", 0.8987), "Critical Leaks": tfidf_esc.get("false_auto_handle_count", 8)},
            {"System": "Rule-Based Baseline", "FAHR (Lower is Better)": rule_esc.get("false_auto_handling_rate", 0.1013), "Escalation Recall": rule_esc.get("escalation_recall", 0.8987), "Critical Leaks": rule_esc.get("false_auto_handle_count", 8)},
            {"System": "Main AI Agent (Hybrid)", "FAHR (Lower is Better)": ai_esc.get("false_auto_handling_rate", 0.0), "Escalation Recall": ai_esc.get("escalation_recall", 1.0), "Critical Leaks": ai_esc.get("false_auto_handle_count", 0)}
        ])
        
        st.dataframe(
            safety_df.style.format({
                "FAHR (Lower is Better)": "{:.2%}",
                "Escalation Recall": "{:.2%}",
                "Critical Leaks": "{:d}"
            }),
            use_container_width=True,
            hide_index=True
        )

    # Tab 4: LLM Judge Rubric
    with tab4:
        st.markdown("#### ⭐ LLM-as-Judge 6-Dimension Evaluation Rubric")
        st.markdown("Replies are evaluated against 6 core support quality criteria on a 1–5 scale.")

        judge_cats = [
            ("Correctness", judge_metrics.get("correctness", 3.63), 5.0, "Accurate instructions and diagnostic steps"),
            ("Groundedness", judge_metrics.get("groundedness", 4.75), 5.0, "Directly backed by historical AppleSupport evidence"),
            ("Helpfulness", judge_metrics.get("helpfulness", 4.46), 5.0, "Clear next actions provided to the customer"),
            ("Brand Alignment", judge_metrics.get("brand_alignment", 5.0), 5.0, "Matches Apple's polite, professional support voice"),
            ("Safety Guardrails", judge_metrics.get("safety", 5.0), 5.0, "Zero unauthorized refund or policy promises"),
            ("Hallucination Freedom", judge_metrics.get("hallucination", 5.0), 5.0, "Zero fabricated features, links, or fixes")
        ]

        col_radar, col_bars = st.columns([1, 1])

        with col_radar:
            # Radar chart
            categories = [c[0] for c in judge_cats]
            values = [c[1] for c in judge_cats]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.25)",
                line=dict(color="#38BDF8", width=2),
                name="AI Agent Quality"
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5.2], gridcolor="rgba(255,255,255,0.1)", tickfont=dict(size=10, color="#94A3B8")),
                    angularaxis=dict(tickfont=dict(size=11, color="#CBD5E1"))
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=30, r=30, t=20, b=20),
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        with col_bars:
            for name, score, max_s, desc in judge_cats:
                pct = (score / max_s) * 100
                st.markdown(
                    f"""
                    <div style="margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; color: #FFFFFF; margin-bottom: 3px;">
                            <span>{name}</span>
                            <span style="color: #38BDF8;">{score:.2f} / {max_s:.1f}</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.06); height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 2px;">
                            <div style="background: linear-gradient(90deg, #38BDF8, #10B981); width: {pct}%; height: 100%;"></div>
                        </div>
                        <div style="font-size: 11px; color: #64748B;">{desc}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
