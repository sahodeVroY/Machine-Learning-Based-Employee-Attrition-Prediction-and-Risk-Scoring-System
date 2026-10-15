"""
app.py
------
Streamlit dashboard for the Employee Attrition Prediction & Risk Scoring System
Palo Alto Networks — HR Analytics Platform
"""

import os, sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.risk_scoring import score_employees, assign_risk_category, RISK_COLOR_MAP
from src.explainability import get_feature_importance, explain_single_employee

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Intelligence | Palo Alto Networks",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark gradient background */
.stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); }

/* Main title */
.main-title {
    font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #f97316, #ef4444, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; padding-bottom: 4px;
}
.sub-title { font-size: 1rem; color: #94a3b8; margin-top: -8px; }

/* KPI cards */
.kpi-card {
    background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px; padding: 20px 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.kpi-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
.kpi-label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-delta { font-size: 0.82rem; margin-top: 4px; }

/* Risk badge */
.badge-low { background: #14532d; color: #4ade80; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.badge-med { background: #451a03; color: #fbbf24; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
.badge-high { background: #450a0a; color: #f87171; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    border-right: 1px solid rgba(148,163,184,0.1);
}

/* Section header */
.section-header {
    font-size: 1.15rem; font-weight: 600; color: #e2e8f0;
    border-left: 4px solid #f97316; padding-left: 12px;
    margin: 1.5rem 0 0.75rem;
}

/* Divider */
.hr { border: none; border-top: 1px solid rgba(148,163,184,0.12); margin: 1.5rem 0; }

/* Employee table */
.risk-high { color: #f87171; font-weight: 600; }
.risk-med  { color: #fbbf24; font-weight: 600; }
.risk-low  { color: #4ade80; font-weight: 600; }

/* Alert / info box */
.info-box {
    background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.3);
    border-radius: 10px; padding: 12px 16px; font-size: 0.9rem; color: #93c5fd;
    margin: 8px 0;
}
.warning-box {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3);
    border-radius: 10px; padding: 12px 16px; font-size: 0.9rem; color: #fca5a5;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: load models / data ───────────────────────────────────────────────
def _run_pipeline_if_needed():
    """Run the training pipeline if model files are missing (e.g. on Streamlit Cloud)."""
    models_dir = os.path.join(ROOT, "models")
    required = ["logistic_regression.pkl", "random_forest.pkl", "xgboost.pkl",
                "scaler.pkl", "encoders.pkl", "feature_cols.pkl", "best_model_name.pkl"]
    missing = [r for r in required if not os.path.exists(os.path.join(models_dir, r))]
    if not missing:
        return True  # All good, nothing to do

    # Some files are missing — run the pipeline automatically
    with st.spinner("🚀 First-time setup: training models… this takes ~2 min, please wait."):
        import subprocess
        pipeline_script = os.path.join(ROOT, "run_pipeline.py")
        result = subprocess.run(
            [sys.executable, pipeline_script],
            capture_output=True, text=True, cwd=ROOT
        )
        if result.returncode != 0:
            st.error(f"Pipeline failed:\n```\n{result.stderr[-3000:]}\n```")
            return False
    return True


@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    if not _run_pipeline_if_needed():
        return None

    models_dir = os.path.join(ROOT, "models")
    lr = joblib.load(os.path.join(models_dir, "logistic_regression.pkl"))
    rf = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
    xgb = joblib.load(os.path.join(models_dir, "xgboost.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
    feature_cols = joblib.load(os.path.join(models_dir, "feature_cols.pkl"))
    best_name = joblib.load(os.path.join(models_dir, "best_model_name.pkl"))
    metrics_path = os.path.join(models_dir, "model_metrics.csv")
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    return {
        "models": {"logistic_regression": lr, "random_forest": rf, "xgboost": xgb},
        "best_name": best_name,
        "best_model": {"logistic_regression": lr, "random_forest": rf, "xgboost": xgb}[best_name],
        "scaler": scaler,
        "encoders": encoders,
        "feature_cols": feature_cols,
        "metrics_df": metrics_df,
    }


@st.cache_data(show_spinner="Processing data…")
def load_and_score_data(_artifacts):
    """Load raw data, apply full preprocessing, score all employees."""
    from src.data_preprocessing import load_raw_data, clean_data, encode_categoricals
    from src.feature_engineering import add_engineered_features

    data_path = os.path.join(ROOT, "data", "raw", "employee_data.csv")
    df_raw = load_raw_data(data_path)
    df = clean_data(df_raw.copy())
    df, _ = encode_categoricals(df)
    df = add_engineered_features(df)

    feature_cols = _artifacts["feature_cols"]
    scaler = _artifacts["scaler"]
    model = _artifacts["best_model"]

    feat_present = [c for c in feature_cols if c in df.columns]
    X_all = df[feat_present]

    # Scale only the columns the scaler knows
    try:
        X_scaled = pd.DataFrame(scaler.transform(X_all), columns=feat_present)
    except Exception:
        X_scaled = X_all.copy()

    risk_df = score_employees(model, X_scaled)

    # Re-attach some descriptive raw columns
    display_cols = ["Department", "JobRole", "Gender", "Age", "MonthlyIncome",
                    "JobSatisfaction", "OverTime", "Attrition"]
    for col in display_cols:
        if col in df_raw.columns:
            risk_df[col] = df_raw[col].values

    risk_df["X_scaled"] = list(X_scaled.values)  # for what-if
    risk_df["EmployeeIndex"] = range(len(risk_df))
    return risk_df, X_scaled, df_raw


def kpi_card(label, value, color="#f97316", delta=None):
    delta_html = f'<div class="kpi-delta" style="color:{color};">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color};">{value}</div>
        {delta_html}
    </div>"""


def risk_badge(category):
    cls = {"Low Risk": "badge-low", "Medium Risk": "badge-med", "High Risk": "badge-high"}.get(category, "badge-low")
    return f'<span class="{cls}">{category}</span>'


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="display:flex; align-items:center; gap:16px; margin-bottom:8px;">
    <div>
        <div class="main-title">🔥 Employee Attrition Intelligence</div>
        <div class="sub-title">Palo Alto Networks · ML-Powered HR Risk Analytics Platform</div>
    </div>
</div>
<div class="hr"></div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════
artifacts = load_artifacts()

if artifacts is None:
    st.error("⚠️ Model artifacts not found. Please run `python run_pipeline.py` first to train and save the models.")
    st.code("python run_pipeline.py", language="bash")
    st.stop()

risk_df, X_scaled_full, df_raw = load_and_score_data(artifacts)
feature_cols = artifacts["feature_cols"]
best_model = artifacts["best_model"]
best_name = artifacts["best_name"]


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 12px 0 4px;">
        <div style="font-size:1.8rem;">🔥</div>
        <div style="font-size:1rem; font-weight:600; color:#e2e8f0;">Attrition Intelligence</div>
        <div style="font-size:0.72rem; color:#64748b; margin-top:2px;">Palo Alto Networks HR Analytics</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🎛️ Filters")

    # Department filter
    depts = ["All"] + sorted(df_raw["Department"].dropna().unique().tolist()) if "Department" in df_raw.columns else ["All"]
    sel_dept = st.selectbox("Department", depts)

    # Job Role filter
    roles = ["All"] + sorted(df_raw["JobRole"].dropna().unique().tolist()) if "JobRole" in df_raw.columns else ["All"]
    sel_role = st.selectbox("Job Role", roles)

    # Age group
    age_bins = {"All": (0, 99), "Under 30": (0, 29), "30–40": (30, 40), "41–50": (41, 50), "50+": (51, 99)}
    sel_age_group = st.selectbox("Age Group", list(age_bins.keys()))

    # Risk level filter
    risk_levels = ["All", "Low Risk", "Medium Risk", "High Risk"]
    sel_risk = st.selectbox("Risk Level", risk_levels)

    st.markdown("---")
    st.markdown("### 🎚️ Risk Thresholds")
    low_thresh = st.slider("Low → Medium boundary", 0.10, 0.50, 0.30, 0.01,
                           help="Probability below this = Low Risk")
    high_thresh = st.slider("Medium → High boundary", 0.40, 0.90, 0.60, 0.01,
                            help="Probability above this = High Risk")

    st.markdown("---")
    st.markdown("### 🤖 Active Model")
    model_choice = st.selectbox("Model", list(artifacts["models"].keys()),
                                index=list(artifacts["models"].keys()).index(best_name))
    active_model = artifacts["models"][model_choice]

    st.markdown("---")
    page = st.radio("📑 Navigate", [
        "📊 Risk Overview",
        "👤 Employee Profiles",
        "🏢 Department Analysis",
        "🔍 Explainability",
        "🎲 What-If Simulator",
        "📈 Model Performance",
    ])

# ── Re-score with selected thresholds & model ────────────────────────────────
@st.cache_data(show_spinner=False)
def rescore(_model, _X_scaled, lt, ht):
    return score_employees(_model, _X_scaled, low_threshold=lt, high_threshold=ht)

scored = rescore(active_model, X_scaled_full, low_thresh, high_thresh)
risk_df["AttritionProbability"] = scored["AttritionProbability"].values
risk_df["RiskCategory"] = scored["RiskCategory"].values

# ── Apply filters ────────────────────────────────────────────────────────────
fdf = risk_df.copy()
if sel_dept != "All" and "Department" in fdf.columns:
    fdf = fdf[fdf["Department"] == sel_dept]
if sel_role != "All" and "JobRole" in fdf.columns:
    fdf = fdf[fdf["JobRole"] == sel_role]
if "Age" in fdf.columns:
    lo, hi = age_bins[sel_age_group]
    fdf = fdf[(fdf["Age"] >= lo) & (fdf["Age"] <= hi)]
if sel_risk != "All":
    fdf = fdf[fdf["RiskCategory"] == sel_risk]


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — RISK OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page == "📊 Risk Overview":
    total = len(fdf)
    high_risk = (fdf["RiskCategory"] == "High Risk").sum()
    med_risk = (fdf["RiskCategory"] == "Medium Risk").sum()
    low_risk = (fdf["RiskCategory"] == "Low Risk").sum()
    avg_prob = fdf["AttritionProbability"].mean() if total > 0 else 0
    actual_rate = fdf["Attrition"].mean() * 100 if "Attrition" in fdf.columns and total > 0 else 0

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Total Employees", f"{total:,}", "#60a5fa"), unsafe_allow_html=True)
    c2.markdown(kpi_card("🔴 High Risk", f"{high_risk:,}", "#ef4444",
                         f"{high_risk/total*100:.1f}% of total" if total else "—"), unsafe_allow_html=True)
    c3.markdown(kpi_card("🟡 Medium Risk", f"{med_risk:,}", "#f59e0b",
                         f"{med_risk/total*100:.1f}% of total" if total else "—"), unsafe_allow_html=True)
    c4.markdown(kpi_card("🟢 Low Risk", f"{low_risk:,}", "#22c55e",
                         f"{low_risk/total*100:.1f}% of total" if total else "—"), unsafe_allow_html=True)
    c5.markdown(kpi_card("Avg. Attrition Prob.", f"{avg_prob*100:.1f}%", "#a78bfa",
                         f"Actual: {actual_rate:.1f}%"), unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    if total == 0:
        st.warning("No employees match the selected filters.")
        st.stop()

    # ── Charts row ──
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = fdf["RiskCategory"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        order = ["High Risk", "Medium Risk", "Low Risk"]
        risk_counts["Risk"] = pd.Categorical(risk_counts["Risk"], categories=order, ordered=True)
        risk_counts = risk_counts.sort_values("Risk")
        fig_pie = px.pie(
            risk_counts, names="Risk", values="Count",
            color="Risk",
            color_discrete_map=RISK_COLOR_MAP,
            hole=0.55,
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13,
                              marker=dict(line=dict(color="#0f172a", width=2)))
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=0, r=0, t=10, b=0), height=320,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Probability Distribution</div>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            fdf, x="AttritionProbability", nbins=40,
            color_discrete_sequence=["#f97316"],
            labels={"AttritionProbability": "Attrition Probability"},
        )
        fig_hist.add_vline(x=low_thresh, line_dash="dash", line_color="#22c55e",
                           annotation_text=f"Low / Med ({low_thresh:.0%})", annotation_font_color="#22c55e")
        fig_hist.add_vline(x=high_thresh, line_dash="dash", line_color="#ef4444",
                           annotation_text=f"Med / High ({high_thresh:.0%})", annotation_font_color="#ef4444")
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", xaxis_title="Attrition Probability",
            yaxis_title="Count", margin=dict(l=0, r=0, t=10, b=0), height=320,
            bargap=0.05,
        )
        fig_hist.update_traces(marker_line_color="#0f172a", marker_line_width=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Risk by department / job role side by side ──
    col_c, col_d = st.columns(2)
    with col_c:
        if "Department" in fdf.columns:
            st.markdown('<div class="section-header">Risk by Department</div>', unsafe_allow_html=True)
            dept_risk = fdf.groupby(["Department", "RiskCategory"]).size().reset_index(name="Count")
            fig_dept = px.bar(dept_risk, x="Department", y="Count", color="RiskCategory",
                              color_discrete_map=RISK_COLOR_MAP, barmode="stack",
                              labels={"Count": "Employees"})
            fig_dept.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", legend_title_text="Risk",
                margin=dict(l=0, r=0, t=10, b=0), height=300,
            )
            st.plotly_chart(fig_dept, use_container_width=True)

    with col_d:
        if "OverTime" in df_raw.columns:
            st.markdown('<div class="section-header">Overtime vs Risk</div>', unsafe_allow_html=True)
            ot_df = fdf.copy()
            ot_df["OverTime"] = ot_df["OverTime"].map({"Yes": "Overtime", "No": "No OT",
                                                        1: "Overtime", 0: "No OT"})
            ot_risk = ot_df.groupby(["OverTime", "RiskCategory"]).size().reset_index(name="Count")
            fig_ot = px.bar(ot_risk, x="OverTime", y="Count", color="RiskCategory",
                            color_discrete_map=RISK_COLOR_MAP, barmode="group",
                            labels={"Count": "Employees", "OverTime": ""})
            fig_ot.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", margin=dict(l=0, r=0, t=10, b=0), height=300,
            )
            st.plotly_chart(fig_ot, use_container_width=True)

    # ── Data export ──
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    export_cols = ["EmployeeIndex", "AttritionProbability", "RiskCategory"] + \
                  [c for c in ["Department", "JobRole", "Age", "Gender", "MonthlyIncome"] if c in fdf.columns]
    csv_data = fdf[export_cols].to_csv(index=False)
    st.download_button("📥 Export Filtered Data (CSV)", data=csv_data,
                       file_name="attrition_risk_report.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — EMPLOYEE PROFILES
# ═══════════════════════════════════════════════════════════════
elif page == "👤 Employee Profiles":
    st.markdown('<div class="section-header">Employee Risk Profiles</div>', unsafe_allow_html=True)

    # Search
    search_id = st.number_input("Employee Index", min_value=0,
                                 max_value=len(risk_df) - 1, value=0, step=1)
    emp = risk_df.iloc[search_id]
    prob = emp["AttritionProbability"]
    risk_cat = emp["RiskCategory"]

    # Profile card
    prob_color = RISK_COLOR_MAP.get(risk_cat, "#ffffff")
    col_info, col_gauge = st.columns([1, 1])

    with col_info:
        info_items = {
            "Department": emp.get("Department", "—"),
            "Job Role": emp.get("JobRole", "—"),
            "Age": emp.get("Age", "—"),
            "Gender": emp.get("Gender", "—"),
            "Monthly Income": f"${emp.get('MonthlyIncome', 0):,.0f}",
            "Overtime": "Yes" if emp.get("OverTime", 0) in [1, "Yes"] else "No",
            "Actual Attrition": "Attrited" if emp.get("Attrition", 0) == 1 else "Stayed",
        }
        rows = "".join([f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    padding:8px 0; border-bottom:1px solid rgba(148,163,184,0.1);">
            <span style="color:#94a3b8; font-size:0.85rem;">{k}</span>
            <span style="color:#e2e8f0; font-weight:500;">{v}</span>
        </div>""" for k, v in info_items.items()])
        st.markdown(f"""
        <div class="kpi-card">
            <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:1px;">Employee #{search_id}</div>
            {rows}
        </div>""", unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            title={"text": f"Attrition Risk<br><span style='font-size:0.85em; color:{prob_color};'>{risk_cat}</span>",
                   "font": {"color": "#e2e8f0", "size": 16}},
            number={"suffix": "%", "font": {"color": prob_color, "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": prob_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,0,0,0)",
                "threshold": {"line": {"color": "#ef4444", "width": 3}, "thickness": 0.8, "value": 60},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
            height=280, margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # SHAP explanation for this employee
    st.markdown('<div class="section-header">Key Risk Factors (SHAP Explanation)</div>', unsafe_allow_html=True)
    feat_present = [c for c in feature_cols if c in X_scaled_full.columns]
    employee_X = X_scaled_full.iloc[[search_id]][feat_present]

    with st.spinner("Computing SHAP values…"):
        try:
            contrib_df = explain_single_employee(active_model, employee_X, feat_present, top_n=10)
            if "Contribution" in contrib_df.columns:
                colors = ["#ef4444" if v > 0 else "#22c55e" for v in contrib_df["Contribution"]]
                fig_contrib = go.Figure(go.Bar(
                    x=contrib_df["Contribution"],
                    y=contrib_df["Feature"],
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.3f}" for v in contrib_df["Contribution"]],
                    textposition="outside",
                ))
                fig_contrib.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", xaxis_title="SHAP Contribution",
                    yaxis={"autorange": "reversed"},
                    height=360, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_contrib, use_container_width=True)
            else:
                # Feature importance fallback
                fig_fi = px.bar(contrib_df, x="Importance", y="Feature", orientation="h",
                                color_discrete_sequence=["#f97316"])
                fig_fi.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", yaxis={"autorange": "reversed"},
                    height=360, margin=dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not compute SHAP: {e}")

    # Scrollable table
    st.markdown('<div class="section-header">All Employees (Filtered)</div>', unsafe_allow_html=True)
    table_cols = ["EmployeeIndex", "AttritionProbability", "RiskCategory"] + \
                 [c for c in ["Department", "JobRole", "Age", "MonthlyIncome"] if c in fdf.columns]
    display_tbl = fdf[table_cols].head(200).copy()
    display_tbl["AttritionProbability"] = (display_tbl["AttritionProbability"] * 100).round(1).astype(str) + "%"
    st.dataframe(display_tbl, use_container_width=True, height=350,
                 column_config={"RiskCategory": st.column_config.TextColumn("Risk", width="small")})

    csv_emp = fdf[table_cols].to_csv(index=False)
    st.download_button("📥 Download Employee Profiles", data=csv_emp,
                       file_name="employee_profiles.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — DEPARTMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif page == "🏢 Department Analysis":
    st.markdown('<div class="section-header">Department & Role Risk Intelligence</div>', unsafe_allow_html=True)

    if "Department" not in fdf.columns:
        st.warning("Department column not available.")
        st.stop()

    # Dept summary table
    dept_summary = fdf.groupby("Department").agg(
        Employees=("AttritionProbability", "count"),
        AvgRisk=("AttritionProbability", "mean"),
        HighRisk=("RiskCategory", lambda x: (x == "High Risk").sum()),
        MedRisk=("RiskCategory", lambda x: (x == "Medium Risk").sum()),
        LowRisk=("RiskCategory", lambda x: (x == "Low Risk").sum()),
    ).reset_index()
    dept_summary["HighRisk%"] = (dept_summary["HighRisk"] / dept_summary["Employees"] * 100).round(1)
    dept_summary["AvgRisk%"] = (dept_summary["AvgRisk"] * 100).round(1)

    # KPIs per dept
    cols = st.columns(len(dept_summary))
    for i, (_, row) in enumerate(dept_summary.iterrows()):
        cols[i].markdown(kpi_card(
            row["Department"],
            f"{row['HighRisk%']}% high risk",
            "#ef4444" if row["HighRisk%"] > 20 else "#22c55e",
            f"{row['Employees']:,} employees · avg {row['AvgRisk%']}%",
        ), unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Average Risk by Department</div>', unsafe_allow_html=True)
        fig_dept_avg = px.bar(
            dept_summary, x="Department", y="AvgRisk%",
            color="AvgRisk%", color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            text="AvgRisk%", labels={"AvgRisk%": "Avg. Attrition Prob (%)"},
        )
        fig_dept_avg.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_dept_avg.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0), height=300,
        )
        st.plotly_chart(fig_dept_avg, use_container_width=True)

    with col_r:
        if "JobRole" in fdf.columns:
            st.markdown('<div class="section-header">High Risk Employees by Job Role</div>', unsafe_allow_html=True)
            role_hr = fdf[fdf["RiskCategory"] == "High Risk"]["JobRole"].value_counts().reset_index()
            role_hr.columns = ["JobRole", "Count"]
            fig_role = px.bar(role_hr, x="Count", y="JobRole", orientation="h",
                              color_discrete_sequence=["#ef4444"],
                              labels={"Count": "High Risk Count", "JobRole": ""})
            fig_role.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", yaxis={"autorange": "reversed"},
                margin=dict(l=0, r=0, t=10, b=0), height=300,
            )
            st.plotly_chart(fig_role, use_container_width=True)

    # Heatmap: dept × job role
    if "JobRole" in fdf.columns:
        st.markdown('<div class="section-header">Risk Heatmap: Department × Job Role</div>', unsafe_allow_html=True)
        heatmap_df = fdf.groupby(["Department", "JobRole"])["AttritionProbability"].mean().reset_index()
        heatmap_pivot = heatmap_df.pivot(index="JobRole", columns="Department", values="AttritionProbability")
        fig_heat = px.imshow(
            heatmap_pivot,
            color_continuous_scale=["#14532d", "#d97706", "#7f1d1d"],
            aspect="auto", text_auto=".0%",
        )
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", height=420,
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Avg Risk"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Summary table
    st.markdown('<div class="section-header">Summary Table</div>', unsafe_allow_html=True)
    st.dataframe(dept_summary, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Explainability":
    st.markdown('<div class="section-header">Global Feature Importance</div>', unsafe_allow_html=True)

    feat_present = [c for c in feature_cols if c in X_scaled_full.columns]

    try:
        fi_df = get_feature_importance(active_model, feat_present, top_n=20)
    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")
        st.stop()

    col_fi, col_info = st.columns([2, 1])
    with col_fi:
        fig_fi = px.bar(
            fi_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#1e3a5f", "#3b82f6", "#f97316"],
            labels={"Importance": "Importance Score", "Feature": ""},
        )
        fig_fi.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", coloraxis_showscale=False,
            height=480, margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>About Feature Importance</b><br><br>
        Shows which attributes most strongly influence the model's attrition predictions.
        Higher values = greater influence on the prediction outcome.
        </div>
        <br>
        <div class="info-box">
        <b>Tree-based models</b> (RF, XGBoost) report mean decrease in impurity.<br><br>
        <b>Logistic Regression</b> reports absolute coefficient magnitudes.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Top 5 Features:**")
        for _, row in fi_df.head(5).iterrows():
            pct = row["Importance"] / fi_df["Importance"].max() * 100
            st.markdown(f"""
            <div style="margin:6px 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                    <span style="color:#e2e8f0;">{row['Feature']}</span>
                    <span style="color:#f97316; font-weight:600;">{row['Importance']:.4f}</span>
                </div>
                <div style="background:#1e293b; border-radius:4px; height:6px; margin-top:3px;">
                    <div style="background:linear-gradient(90deg,#3b82f6,#f97316); width:{pct:.0f}%; height:6px; border-radius:4px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # SHAP global summary (sample-based)
    st.markdown('<div class="section-header">SHAP Summary (Sample of 200 Employees)</div>', unsafe_allow_html=True)
    with st.spinner("Computing SHAP values for sample…"):
        try:
            import shap
            shap_vals, explainer, X_sample = None, None, None

            X_scaled_sample = X_scaled_full[feat_present].sample(min(200, len(X_scaled_full)), random_state=42)
            try:
                tree_exp = shap.TreeExplainer(active_model)
                sv = tree_exp.shap_values(X_scaled_sample)
                if isinstance(sv, list):
                    sv = sv[1]
            except Exception:
                bg = shap.sample(X_scaled_full[feat_present], 50)
                k_exp = shap.KernelExplainer(active_model.predict_proba, bg)
                sv = k_exp.shap_values(X_scaled_sample)
                if isinstance(sv, list):
                    sv = sv[1]

            # Mean absolute SHAP per feature
            mean_shap = pd.DataFrame({
                "Feature": feat_present,
                "MeanAbsSHAP": np.abs(sv).mean(axis=0),
            }).sort_values("MeanAbsSHAP", ascending=False).head(20)

            fig_shap = px.bar(
                mean_shap.sort_values("MeanAbsSHAP"), x="MeanAbsSHAP", y="Feature", orientation="h",
                color="MeanAbsSHAP", color_continuous_scale=["#1e3a5f", "#8b5cf6", "#ef4444"],
                labels={"MeanAbsSHAP": "Mean |SHAP|", "Feature": ""},
            )
            fig_shap.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", coloraxis_showscale=False,
                height=420, margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP computation failed: {e}. Showing feature importance above.")


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════════════
elif page == "🎲 What-If Simulator":
    st.markdown('<div class="section-header">What-If Scenario Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Modify an employee's attributes below and see how the attrition probability changes in real time.
    </div>
    """, unsafe_allow_html=True)

    search_id = st.selectbox("Select Employee", range(len(risk_df)), index=0,
                              format_func=lambda i: f"Employee #{i} · {risk_df.iloc[i]['RiskCategory']}")

    emp_row = X_scaled_full.iloc[[search_id]].copy()
    feat_present = [c for c in feature_cols if c in X_scaled_full.columns]
    emp_row = emp_row[feat_present]

    # Get original raw values for editing
    raw_row = df_raw.iloc[search_id]
    baseline_prob = active_model.predict_proba(emp_row)[:, 1][0]

    st.markdown(f"""
    <div class="kpi-card" style="margin-bottom:16px;">
        <div class="kpi-label">Baseline Attrition Probability</div>
        <div class="kpi-value" style="color:{RISK_COLOR_MAP.get(assign_risk_category(baseline_prob, low_thresh, high_thresh), '#fff')};">
            {baseline_prob*100:.1f}%
        </div>
        <div class="kpi-delta" style="color:#94a3b8;">{assign_risk_category(baseline_prob, low_thresh, high_thresh)}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Adjust Attributes:**")
    col1, col2, col3 = st.columns(3)

    # Key sliders based on feature importance
    with col1:
        job_sat = st.slider("Job Satisfaction (1–4)", 1, 4,
                             int(raw_row.get("JobSatisfaction", 2)), key="js")
        env_sat = st.slider("Environment Satisfaction (1–4)", 1, 4,
                              int(raw_row.get("EnvironmentSatisfaction", 2)), key="es")
        wlb = st.slider("Work-Life Balance (1–4)", 1, 4,
                         int(raw_row.get("WorkLifeBalance", 2)), key="wlb")
    with col2:
        overtime = st.selectbox("Overtime", ["No", "Yes"],
                                 index=1 if raw_row.get("OverTime", 0) in [1, "Yes"] else 0)
        yrs_promo = st.slider("Years Since Last Promotion", 0, 15,
                               int(raw_row.get("YearsSinceLastPromotion", 1)), key="ysp")
        salary_hike = st.slider("% Salary Hike", 10, 25,
                                 int(raw_row.get("PercentSalaryHike", 12)), key="sh")
    with col3:
        training = st.slider("Training Times Last Year", 0, 6,
                              int(raw_row.get("TrainingTimesLastYear", 2)), key="tt")
        job_inv = st.slider("Job Involvement (1–4)", 1, 4,
                             int(raw_row.get("JobInvolvement", 2)), key="ji")
        dist_home = st.slider("Distance From Home", 1, 30,
                               int(raw_row.get("DistanceFromHome", 5)), key="dh")

    # Build modified row from raw-space edits, then re-scale
    from src.data_preprocessing import load_raw_data, clean_data, encode_categoricals
    from src.feature_engineering import add_engineered_features

    scaler = artifacts["scaler"]

    @st.cache_data(show_spinner=False)
    def get_modified_prob(_model, _scaler, _feat_cols, orig_idx,
                          js, es, wlb, ot, ysp, sh, tt, ji, dh, lt, ht):
        # Load full df + preprocess
        data_path = os.path.join(ROOT, "data", "raw", "employee_data.csv")
        df_mod = load_raw_data(data_path)
        df_mod = clean_data(df_mod.copy())
        df_mod, _ = encode_categoricals(df_mod)
        df_mod = add_engineered_features(df_mod)

        row = df_mod.iloc[[orig_idx]].copy()

        # Apply changes
        changes = {
            "JobSatisfaction": js, "EnvironmentSatisfaction": es,
            "WorkLifeBalance": wlb, "OverTime": 1 if ot == "Yes" else 0,
            "YearsSinceLastPromotion": ysp, "PercentSalaryHike": sh,
            "TrainingTimesLastYear": tt, "JobInvolvement": ji, "DistanceFromHome": dh,
        }
        for col, val in changes.items():
            if col in row.columns:
                row[col] = val

        # Update engineered features
        if "MonthlyIncome" in row.columns and "TotalWorkingYears" in row.columns:
            row["IncomeExperienceRatio"] = row["MonthlyIncome"] / (row["TotalWorkingYears"] + 1)
        if "YearsSinceLastPromotion" in row.columns:
            row["PromotionDelayFlag"] = int(row["YearsSinceLastPromotion"].iloc[0] >= 4)
        eng_cols = ["JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
                    "WorkLifeBalance", "JobInvolvement"]
        present_eng = [c for c in eng_cols if c in row.columns]
        row["EngagementScore"] = row[present_eng].mean(axis=1)
        row["WorkloadStressFlag"] = row.get("OverTime", pd.Series([0])).iloc[0]

        feat_p = [c for c in _feat_cols if c in row.columns]
        X_mod = row[feat_p]
        try:
            X_scaled_mod = pd.DataFrame(_scaler.transform(X_mod), columns=feat_p)
        except Exception:
            X_scaled_mod = X_mod
        new_prob = _model.predict_proba(X_scaled_mod)[:, 1][0]
        return new_prob

    new_prob = get_modified_prob(
        active_model, scaler, feature_cols, search_id,
        job_sat, env_sat, wlb, overtime, yrs_promo, salary_hike,
        training, job_inv, dist_home, low_thresh, high_thresh
    )
    new_risk = assign_risk_category(new_prob, low_thresh, high_thresh)
    delta = new_prob - baseline_prob

    # Before/After gauge
    col_b, col_a = st.columns(2)
    with col_b:
        bcolor = RISK_COLOR_MAP.get(assign_risk_category(baseline_prob, low_thresh, high_thresh), "#fff")
        fig_b = go.Figure(go.Indicator(
            mode="gauge+number", value=baseline_prob * 100,
            title={"text": "Before", "font": {"color": "#94a3b8"}},
            number={"suffix": "%", "font": {"color": bcolor, "size": 38}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": bcolor},
                   "steps": [{"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                              {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                              {"range": [60, 100], "color": "rgba(239,68,68,0.15)"}],
                   "bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)"},
        ))
        fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                            height=240, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_b, use_container_width=True)

    with col_a:
        acolor = RISK_COLOR_MAP.get(new_risk, "#fff")
        fig_a = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=new_prob * 100,
            title={"text": f"After · <span style='color:{acolor};'>{new_risk}</span>",
                   "font": {"color": "#e2e8f0"}},
            number={"suffix": "%", "font": {"color": acolor, "size": 38}},
            delta={"reference": baseline_prob * 100,
                   "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#22c55e"}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": acolor},
                   "steps": [{"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                              {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                              {"range": [60, 100], "color": "rgba(239,68,68,0.15)"}],
                   "bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)"},
        ))
        fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                            height=240, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_a, use_container_width=True)

    sign = "🔴 +" if delta > 0 else "🟢 "
    st.markdown(f"""
    <div class="{'warning-box' if delta > 0 else 'info-box'}">
        {sign}{abs(delta)*100:.2f}% change in attrition probability
        {'— Interventions needed!' if delta > 0 else '— Scenario improves retention outlook.'}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">Model Evaluation Metrics</div>', unsafe_allow_html=True)

    metrics_df = artifacts["metrics_df"]
    if metrics_df is None:
        st.warning("No metrics file found. Run `python run_pipeline.py` first.")
        st.stop()

    # Highlight best model
    best = metrics_df.iloc[0]["Model"]
    st.markdown(f"""
    <div class="info-box">🏆 Best performing model: <b>{best}</b> (by ROC-AUC)</div>
    """, unsafe_allow_html=True)

    # KPI row for best model
    bm = metrics_df.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Accuracy", f"{bm['Accuracy']*100:.1f}%", "#60a5fa"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Precision", f"{bm['Precision']*100:.1f}%", "#a78bfa"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Recall", f"{bm['Recall']*100:.1f}%", "#34d399"), unsafe_allow_html=True)
    c4.markdown(kpi_card("F1-Score", f"{bm['F1-Score']*100:.1f}%", "#f59e0b"), unsafe_allow_html=True)
    c5.markdown(kpi_card("ROC-AUC", f"{bm['ROC-AUC']*100:.1f}%", "#f97316"), unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Metrics comparison chart
    st.markdown('<div class="section-header">All Models — Metrics Comparison</div>', unsafe_allow_html=True)
    metrics_long = metrics_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
                                    var_name="Metric", value_name="Score")
    fig_comp = px.bar(
        metrics_long, x="Metric", y="Score", color="Model", barmode="group",
        color_discrete_sequence=["#3b82f6", "#f97316", "#8b5cf6"],
        text_auto=".3f",
    )
    fig_comp.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", yaxis_range=[0, 1.05],
        height=360, margin=dict(l=0, r=0, t=10, b=0),
    )
    fig_comp.update_traces(textposition="outside")
    st.plotly_chart(fig_comp, use_container_width=True)

    # Full metrics table
    st.markdown('<div class="section-header">Full Metrics Table</div>', unsafe_allow_html=True)
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    <b>Active Model:</b> {model_choice}<br>
    Using this model for all risk scoring and predictions across the dashboard.
    You can change the active model in the sidebar.
    </div>
    """, unsafe_allow_html=True)
