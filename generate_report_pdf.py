"""
generate_report_pdf.py
----------------------
Generates a professional, styled PDF research paper for the
Employee Attrition Prediction project.

Usage:
    python generate_report_pdf.py

Output: reports/research_paper.pdf
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
from reportlab.lib.colors import HexColor

# ── Color Palette ─────────────────────────────────────────────────────
DARK_BG    = HexColor("#0f172a")
ORANGE     = HexColor("#f97316")
SLATE      = HexColor("#64748b")
LIGHT_TEXT = HexColor("#1e293b")
ACCENT     = HexColor("#3b82f6")
RED        = HexColor("#ef4444")
GREEN      = HexColor("#22c55e")
AMBER      = HexColor("#f59e0b")
HEADER_BG  = HexColor("#1e293b")
ROW_ALT    = HexColor("#f1f5f9")
TABLE_HDR  = HexColor("#0f172a")

OUT_PATH   = os.path.join(os.path.dirname(__file__), "reports", "research_paper.pdf")

# ── Page setup ────────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", fontName="Helvetica-Bold", fontSize=22,
        textColor=colors.white, leading=32, alignment=TA_CENTER, spaceAfter=6,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle", fontName="Helvetica", fontSize=11,
        textColor=HexColor("#94a3b8"), leading=16, alignment=TA_CENTER,
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", fontName="Helvetica", fontSize=9.5,
        textColor=HexColor("#cbd5e1"), leading=14, alignment=TA_CENTER,
    )
    styles["h1"] = ParagraphStyle(
        "h1", fontName="Helvetica-Bold", fontSize=15, textColor=ORANGE,
        spaceBefore=18, spaceAfter=6, leading=20, borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "h2", fontName="Helvetica-Bold", fontSize=12, textColor=LIGHT_TEXT,
        spaceBefore=12, spaceAfter=4, leading=16,
    )
    styles["h3"] = ParagraphStyle(
        "h3", fontName="Helvetica-BoldOblique", fontSize=10.5, textColor=ACCENT,
        spaceBefore=8, spaceAfter=3, leading=14,
    )
    styles["body"] = ParagraphStyle(
        "body", fontName="Helvetica", fontSize=10, textColor=LIGHT_TEXT,
        leading=15, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    styles["abstract"] = ParagraphStyle(
        "abstract", fontName="Helvetica", fontSize=9.5, textColor=HexColor("#334155"),
        leading=15, alignment=TA_JUSTIFY, spaceAfter=6,
        leftIndent=20, rightIndent=20,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", fontName="Helvetica", fontSize=10, textColor=LIGHT_TEXT,
        leading=15, leftIndent=18, firstLineIndent=-10, spaceAfter=3,
    )
    styles["caption"] = ParagraphStyle(
        "caption", fontName="Helvetica-Oblique", fontSize=8.5, textColor=SLATE,
        leading=12, alignment=TA_CENTER, spaceAfter=8,
    )
    styles["finding"] = ParagraphStyle(
        "finding", fontName="Helvetica", fontSize=10, textColor=LIGHT_TEXT,
        leading=15, alignment=TA_JUSTIFY,
        leftIndent=16, borderPad=6,
    )
    styles["code"] = ParagraphStyle(
        "code", fontName="Courier", fontSize=8.5, textColor=HexColor("#1e293b"),
        leading=13, leftIndent=16, spaceAfter=4, spaceBefore=4,
        backColor=HexColor("#f8fafc"),
    )
    styles["ref"] = ParagraphStyle(
        "ref", fontName="Helvetica", fontSize=9, textColor=SLATE,
        leading=13, leftIndent=18, spaceAfter=3,
    )
    return styles


def cover_page(styles):
    story = []
    story.append(Spacer(1, 1.6 * cm))
    # Top rule
    story.append(HRFlowable(width="100%", thickness=3, color=ORANGE, spaceAfter=0))
    # Dark cover block (simulated via table)
    cover_data = [[
        Paragraph("Machine Learning-Based<br/>Employee Attrition Prediction<br/>and Risk Scoring System",
                  styles["cover_title"]),
    ]]
    cover_tbl = Table(cover_data, colWidths=[PAGE_W - 2 * MARGIN])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), DARK_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 28),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
    ]))
    story.append(cover_tbl)
    story.append(HRFlowable(width="100%", thickness=3, color=ORANGE, spaceBefore=0, spaceAfter=20))

    story.append(Paragraph("Palo Alto Networks — HR Analytics Platform", styles["cover_subtitle"]))
    story.append(Spacer(1, 0.6 * cm))

    # Meta table
    meta = [
        ["Author",   "Sahodev Chandro Ray"],
        ["Program",  "Computer Science and Engineering"],
        ["Date",     "October 2026"],
        ["Status",   "Completed"],
        ["Dataset",  "1,470 Employee Records · 35 Features"],
        ["Models",   "Logistic Regression · Random Forest · XGBoost"],
        ["Best ROC-AUC", "81.0% (Logistic Regression)"],
    ]
    meta_tbl = Table(meta, colWidths=[5 * cm, 9 * cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",   (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",  (0, 0), (0, -1), ORANGE),
        ("TEXTCOLOR",  (1, 0), (1, -1), LIGHT_TEXT),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 1.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#e2e8f0")))
    story.append(PageBreak())
    return story


def section_rule(story):
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#e2e8f0"), spaceAfter=4))


def metric_table(styles):
    """Model performance comparison table."""
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rows = [
        ["Logistic Regression ⭐", "73.8%", "35.0%", "74.5%", "47.6%", "81.0%"],
        ["XGBoost",               "87.1%", "69.6%", "34.0%", "45.7%", "80.7%"],
        ["Random Forest",         "87.1%", "65.5%", "40.4%", "45.7%", "78.9%"],
    ]
    col_w = [5.2 * cm, 2.2 * cm, 2.2 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm]
    data = [headers] + rows
    tbl = Table(data, colWidths=col_w)
    style = TableStyle([
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ALIGN",         (1, 1), (-1, -1), "CENTER"),
        ("ALIGN",         (0, 1), (0, -1), "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ROW_ALT]),
        # Highlight best row
        ("TEXTCOLOR",     (5, 1), (5, 1), ORANGE),
        ("FONTNAME",      (5, 1), (5, 1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (3, 1), (3, 1), GREEN),
        ("FONTNAME",      (3, 1), (3, 1), "Helvetica-Bold"),
        # Grid
        ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ])
    tbl.setStyle(style)
    return tbl


def feature_importance_table(styles):
    headers = ["Rank", "Feature", "Importance Score", "Interpretation"]
    rows = [
        ["1", "MonthlyIncome",          "0.7462", "Lower compensation → higher risk"],
        ["2", "YearsWithCurrManager",   "0.5417", "Manager relationship stability"],
        ["3", "NumCompaniesWorked",      "0.5250", "Prior job-hopping behavior"],
        ["4", "YearsAtCompany",          "0.5008", "Early/late tenure risk"],
        ["5", "Department",              "0.4756", "Role-specific environment"],
        ["6", "EngagementScore",         "0.4044", "Composite satisfaction signal"],
        ["7", "OverTime / Workload",     "0.3902", "Workload pressure indicator"],
        ["8", "MaritalStatus",           "0.3868", "Life circumstance factor"],
        ["9", "YearsInCurrentRole",      "0.3806", "Role stagnation signal"],
        ["10","IncomeExperienceRatio",   "0.3640", "Underpaid relative to tenure"],
    ]
    col_w = [1.2 * cm, 4.5 * cm, 2.8 * cm, 7 * cm]
    data = [headers] + rows
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ("ALIGN",         (2, 1), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        # Highlight top feature
        ("TEXTCOLOR",     (1, 1), (2, 1), ORANGE),
        ("FONTNAME",      (1, 1), (2, 1), "Helvetica-Bold"),
    ]))
    return tbl


def risk_dist_table(styles):
    headers = ["Risk Category", "Count", "Percentage", "Interpretation"]
    rows = [
        ["🔴 High Risk",   "351", "23.9%", "Priority retention intervention"],
        ["🟡 Medium Risk", "377", "25.6%", "Monitoring and proactive engagement"],
        ["🟢 Low Risk",    "742", "50.5%", "Standard workforce management"],
        ["Total",          "1,470", "100%", "Complete Palo Alto Networks dataset"],
    ]
    col_w = [3.8 * cm, 2.0 * cm, 2.5 * cm, 7.2 * cm]
    data = [headers] + rows
    tbl = Table(data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        ("ALIGN",         (1, 0), (2, -1), "CENTER"),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("TEXTCOLOR",     (0, 1), (0, 1), RED),
        ("TEXTCOLOR",     (0, 2), (0, 2), AMBER),
        ("TEXTCOLOR",     (0, 3), (0, 3), GREEN),
        ("FONTNAME",      (0, 4), (-1, 4), "Helvetica-Bold"),
        ("BACKGROUND",    (0, 4), (-1, 4), HexColor("#f1f5f9")),
        ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    return tbl


def finding_box(title, body, styles, color=ORANGE):
    data = [[
        Paragraph(f"<b>{title}</b><br/>{body}", styles["finding"])
    ]]
    tbl = Table(data, colWidths=[PAGE_W - 2 * MARGIN])
    tbl.setStyle(TableStyle([
        ("LEFTBORDER",    (0, 0), (-1, -1), 4, color),
        ("BACKGROUND",    (0, 0), (-1, -1), HexColor("#f8fafc")),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    return tbl


def build_pdf():
    s = build_styles()
    story = []

    # ─── COVER ────────────────────────────────────────────────────────
    story += cover_page(s)

    # ─── ABSTRACT ─────────────────────────────────────────────────────
    story.append(Paragraph("Abstract", s["h1"]))
    section_rule(story)
    story.append(Paragraph(
        "Employee attrition represents a significant operational and financial challenge for technology "
        "organizations. This paper presents the development and evaluation of a machine learning-based "
        "attrition prediction and risk scoring system designed for Palo Alto Networks. The system applies "
        "a multi-model classification pipeline — comprising Logistic Regression, Random Forest, and "
        "XGBoost — to a 1,470-record employee dataset, producing employee-level attrition probability "
        "estimates and interpretable risk categories. The best-performing model achieves a ROC-AUC of "
        "<b>0.810</b> with a recall of <b>74.5%</b> on attrition cases, demonstrating meaningful "
        "predictive capability beyond chance. Results are surfaced through an interactive Streamlit "
        "dashboard enabling HR stakeholders to explore individual employee risk profiles, department-level "
        "patterns, and model explanations using SHAP values and feature importance. The system is designed "
        "to support proactive, data-driven retention planning rather than reactive workforce management.",
        s["abstract"]
    ))

    # ─── 1. INTRODUCTION ──────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("1. Introduction", s["h1"]))
    section_rule(story)

    story.append(Paragraph("1.1 Background", s["h2"]))
    story.append(Paragraph(
        "Employee attrition — the voluntary or involuntary departure of employees from an organization — "
        "imposes substantial direct and indirect costs. Tangible costs include recruitment, onboarding, "
        "and training; intangible costs include lost institutional knowledge, reduced team cohesion, and "
        "productivity disruption. For technology firms like Palo Alto Networks, where specialized talent "
        "is critical and competitive labor markets intensify turnover risk, the ability to anticipate "
        "and respond to attrition risk proactively is a strategic priority.",
        s["body"]
    ))
    story.append(Paragraph(
        "Traditional HR analytics has largely been retrospective: reporting on attrition rates after "
        "the fact, or identifying patterns in historical exit data. This approach provides limited time "
        "for intervention. A predictive system that identifies employees with elevated departure risk in "
        "advance enables HR teams to act before attrition occurs — through retention conversations, "
        "compensation adjustments, career development support, or targeted engagement programs.",
        s["body"]
    ))

    story.append(Paragraph("1.2 Problem Statement", s["h2"]))
    story.append(Paragraph("Palo Alto Networks faces three interconnected attrition challenges:", s["body"]))
    for item in [
        "<b>Unanticipated departures</b> — Resignations that arrive without warning, leaving insufficient time for knowledge transfer or counter-offers.",
        "<b>Loss of critical talent</b> — High-performing or specialized employees whose departure is disproportionately disruptive.",
        "<b>Reactive retention</b> — Counter-offer strategies deployed after an employee has already decided to leave.",
    ]:
        story.append(Paragraph(f"• {item}", s["bullet"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("1.3 Objectives", s["h2"]))
    for obj in [
        "Develop a machine learning pipeline that classifies employees as likely to attrite or remain.",
        "Generate calibrated attrition probability estimates for each employee.",
        "Assign employees to interpretable risk categories: Low, Medium, and High.",
        "Identify the features most strongly associated with predicted attrition.",
        "Provide an interactive dashboard enabling HR stakeholders to explore findings.",
    ]:
        story.append(Paragraph(f"• {obj}", s["bullet"]))

    # ─── 2. DATASET ───────────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("2. Dataset", s["h1"]))
    section_rule(story)

    story.append(Paragraph("2.1 Overview", s["h2"]))
    story.append(Paragraph(
        "The dataset contains employee-level records for Palo Alto Networks personnel — <b>1,470 records</b> "
        "and <b>35 attributes</b> spanning demographic, compensation, satisfaction, career, workload, and "
        "organizational dimensions. The target variable <i>Attrition</i> is binary (1 = left, 0 = stayed). "
        "The dataset exhibits class imbalance: approximately <b>16.1%</b> of employees attrited (237 records) "
        "versus 83.9% who remained (1,233 records).",
        s["body"]
    ))

    story.append(Paragraph("2.2 Selected Key Features", s["h2"]))
    feat_brief = [
        ("MonthlyIncome", "Monthly salary — top attrition predictor"),
        ("OverTime", "Overtime indicator — workload proxy"),
        ("JobSatisfaction", "Self-rated job satisfaction (1–4)"),
        ("YearsSinceLastPromotion", "Career advancement recency"),
        ("YearsWithCurrManager", "Manager relationship duration"),
        ("NumCompaniesWorked", "Prior employer count — mobility history"),
        ("EnvironmentSatisfaction", "Workplace environment rating (1–4)"),
        ("WorkLifeBalance", "Work-life balance self-rating (1–4)"),
    ]
    feat_table = Table(
        [["Feature", "Description"]] + feat_brief,
        colWidths=[4.5 * cm, 11 * cm]
    )
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",  (0, 1), (0, -1), ACCENT),
        ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(feat_table)

    # ─── 3. METHODOLOGY ───────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("3. Methodology", s["h1"]))
    section_rule(story)

    story.append(Paragraph("3.1 Data Preprocessing", s["h2"]))
    story.append(Paragraph(
        "The preprocessing pipeline applied the following steps: <b>(1) Data Cleaning</b> — removed "
        "constant columns (EmployeeCount, EmployeeNumber, Over18, StandardHours) and converted "
        "categorical targets to binary integers. <b>(2) Label Encoding</b> — all remaining string "
        "columns encoded via LabelEncoder, appropriate for the tree-based models used. "
        "<b>(3) Standard Scaling</b> — all numerical features standardized to zero mean and unit "
        "variance using StandardScaler, applied independently to train and test sets. "
        "<b>(4) Stratified Train-Test Split</b> — 80% training (1,176 records) / 20% test (294 records) "
        "with class-ratio preservation. <b>(5) SMOTE</b> — Synthetic Minority Oversampling Technique "
        "applied after splitting to the training set only, balancing classes to 988 records each "
        "(post-SMOTE training size: 1,976 records).",
        s["body"]
    ))

    story.append(Paragraph("3.2 Feature Engineering", s["h2"]))
    story.append(Paragraph("Four composite features were constructed beyond the raw variables:", s["body"]))
    eng_feats = [
        ("IncomeExperienceRatio", "MonthlyIncome / (TotalWorkingYears + 1)", "Compensation relative to experience — underpaid tenure signals risk"),
        ("PromotionDelayFlag",    "1 if YearsSinceLastPromotion ≥ 4",         "Flags prolonged career stagnation"),
        ("EngagementScore",       "Mean of 5 satisfaction dimensions",         "JobSatisfaction, EnvSatisfaction, RelSatisfaction, WLB, JobInvolvement"),
        ("WorkloadStressFlag",    "1 if OverTime = Yes",                       "Explicit workload pressure indicator derived from OverTime"),
    ]
    eng_tbl = Table(
        [["Feature", "Formula / Logic", "Rationale"]] + eng_feats,
        colWidths=[4.0 * cm, 5.0 * cm, 6.5 * cm]
    )
    eng_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 8.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("FONTNAME",   (0, 1), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR",  (0, 1), (0, -1), ORANGE),
        ("GRID",       (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(eng_tbl)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("3.3 Model Development", s["h2"]))
    for name, desc in [
        ("Logistic Regression",
         "Linear classifier providing an interpretable baseline. Trained with class_weight='balanced' and "
         "max_iter=1000. Coefficients provide directional insight into feature-attrition relationships."),
        ("Random Forest",
         "Bagged ensemble of 300 decision trees. Trained with class_weight='balanced', min_samples_leaf=2. "
         "Captures non-linear relationships and feature interactions. Provides Gini-based feature importance."),
        ("XGBoost",
         "Gradient-boosted ensemble with n_estimators=400, learning_rate=0.05, max_depth=6. "
         "scale_pos_weight set to inverse class ratio for imbalance handling."),
    ]:
        story.append(Paragraph(f"<b>{name}</b>: {desc}", s["body"]))

    story.append(Paragraph("3.4 Risk Scoring Framework", s["h2"]))
    story.append(Paragraph(
        "Attrition probabilities are mapped to three risk categories using configurable thresholds. "
        "Default thresholds are <b>30%</b> (Low/Medium boundary) and <b>60%</b> (Medium/High boundary), "
        "adjustable through the dashboard interface.",
        s["body"]
    ))
    risk_schema = Table(
        [["Probability Range", "Risk Category", "HR Action Priority"]],
        colWidths=[5 * cm, 4 * cm, 6.5 * cm]
    )
    risk_schema.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HDR),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(risk_schema)
    for rng, cat, pri, col in [
        ("p < 30%",      "🟢 Low Risk",    "Standard workforce management", GREEN),
        ("30% ≤ p ≤ 60%","🟡 Medium Risk", "Proactive engagement & monitoring", AMBER),
        ("p > 60%",      "🔴 High Risk",   "Immediate retention intervention", RED),
    ]:
        row_data = [[rng, cat, pri]]
        row_tbl = Table(row_data, colWidths=[5 * cm, 4 * cm, 6.5 * cm])
        row_tbl.setStyle(TableStyle([
            ("FONTNAME",  (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE",  (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (1, 0), (1, 0), col),
            ("FONTNAME",  (1, 0), (1, 0), "Helvetica-Bold"),
            ("GRID",      (0, 0), (-1, -1), 0.3, HexColor("#e2e8f0")),
            ("TOPPADDING",(0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1), 4),
            ("LEFTPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(row_tbl)

    # ─── 4. RESULTS ───────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Results", s["h1"]))
    section_rule(story)

    story.append(Paragraph("4.1 Model Evaluation", s["h2"]))
    story.append(Paragraph(
        "All three models were evaluated on the held-out test set (294 records) using five classification metrics. "
        "The best model is selected by ROC-AUC — the metric most robust to class imbalance.",
        s["body"]
    ))
    story.append(metric_table(s))
    story.append(Paragraph(
        "Caption: ⭐ = Selected model. Highlighted values — ROC-AUC (orange) and Recall (green) — "
        "are the primary selection criteria.",
        s["caption"]
    ))
    story.append(Paragraph(
        "<b>Model Selection Rationale:</b> Logistic Regression was selected as the final model based on its "
        "superior ROC-AUC (81.0%) and highest Recall (74.5%). In attrition prediction, recall is the most "
        "critical operational metric — missing an at-risk employee (false negative) is typically more costly "
        "than an unnecessary retention conversation (false positive). The ROC-AUC of 81.0% indicates strong "
        "discrimination between attrition and retention cases across all probability thresholds.",
        s["body"]
    ))

    story.append(Paragraph("4.2 Risk Distribution", s["h2"]))
    story.append(Paragraph(
        "Applying the selected model to all 1,470 employees with default thresholds (30% / 60%):",
        s["body"]
    ))
    story.append(risk_dist_table(s))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "The model identifies <b>351 employees (23.9%)</b> as High Risk — a focused, actionable cohort "
        "for targeted HR retention interventions.",
        s["body"]
    ))

    story.append(Paragraph("4.3 Top Attrition Predictors (Feature Importance)", s["h2"]))
    story.append(Paragraph(
        "Feature importance from the trained Logistic Regression model (absolute coefficient magnitude). "
        "The top feature's score is normalized to 1.0.",
        s["body"]
    ))
    story.append(feature_importance_table(s))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("4.4 Key Findings", s["h2"]))
    findings = [
        ("Finding 1 — Compensation is the Dominant Predictor",
         "MonthlyIncome is the single most influential predictor (importance: 0.7462). Employees with "
         "lower monthly income relative to their cohort exhibit substantially elevated predicted attrition "
         "risk. Compensation competitiveness is identified as the primary HR lever.",
         ORANGE),
        ("Finding 2 — Manager Relationship Quality Matters",
         "YearsWithCurrManager ranks second (0.5417). Both very short tenure (recent disruption) and "
         "extended tenure without advancement contribute to elevated risk patterns.",
         ACCENT),
        ("Finding 3 — Job Mobility History Predicts Future Mobility",
         "NumCompaniesWorked ranks third (0.5250). Employees with a history of changing employers are "
         "statistically more likely to do so again — reinforcing behavioral alignment in hiring.",
         ACCENT),
        ("Finding 4 — Tenure and Stagnation are Compounding Risk Factors",
         "YearsAtCompany (0.5008) and YearsInCurrentRole (0.3806) suggest both early attrition risk "
         "(adjustment phase) and late-tenure risk (role stagnation without advancement).",
         AMBER),
        ("Finding 5 — Overtime and Low Engagement Compound Risk",
         "WorkloadStressFlag (0.3902) and EngagementScore (0.4044) both appear in the top 10. Employees "
         "simultaneously experiencing high workload and low satisfaction represent a compounding, "
         "priority risk profile.",
         RED),
    ]
    for title, body, col in findings:
        story.append(finding_box(title, body, s, col))
        story.append(Spacer(1, 3 * mm))

    # ─── 5. DISCUSSION ────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("5. Discussion", s["h1"]))
    section_rule(story)

    story.append(Paragraph("5.1 Practical Implications for HR", s["h2"]))
    story.append(Paragraph(
        "The system's output enables HR teams to move from broad workforce statistics to targeted, "
        "evidence-based retention strategies:",
        s["body"]
    ))
    for item in [
        "<b>High-Risk employees (n=351):</b> Priority candidates for immediate retention conversations, "
        "compensation reviews, and engagement check-ins — particularly those flagged for low income, "
        "low engagement, and overtime patterns.",
        "<b>Department-level patterns:</b> The heatmap visualization identifies which departments and "
        "job roles have disproportionate concentrations of high-risk employees, enabling targeted "
        "resource allocation.",
        "<b>What-If Scenario Analysis:</b> HR partners can model the expected change in a specific "
        "employee's attrition probability for hypothetical interventions — providing a quantified "
        "basis for retention ROI estimates.",
    ]:
        story.append(Paragraph(f"• {item}", s["bullet"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("5.2 Limitations", s["h2"]))
    for lim in [
        "Feature importance does not establish causation; unmeasured confounders (team dynamics, external "
        "offers) may also influence attrition.",
        "The model is trained on a historical snapshot dataset. Workforce conditions evolve and require "
        "periodic model re-validation and retraining.",
        "Satisfaction and engagement variables are self-reported and subject to social desirability bias.",
        "Default risk thresholds (30% / 60%) should be calibrated to organizational intervention "
        "capacity and risk tolerance.",
    ]:
        story.append(Paragraph(f"• {lim}", s["bullet"]))
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("5.3 Responsible Use", s["h2"]))
    story.append(finding_box(
        "Important Notice",
        "This system is designed as a decision-support tool, not an automated personnel decision system. "
        "Model predictions must be reviewed by qualified HR professionals, must not be the sole basis for "
        "employment-related decisions, and should be audited for differential performance across protected "
        "demographic groups.",
        s, AMBER
    ))

    # ─── 6. CONCLUSION ────────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("6. Conclusion", s["h1"]))
    section_rule(story)
    story.append(Paragraph(
        "This paper presented a machine learning-based employee attrition prediction and risk scoring "
        "system for Palo Alto Networks. Through a structured pipeline encompassing data preprocessing, "
        "feature engineering, multi-model training, and SHAP-based explainability, the system produces "
        "actionable, employee-level attrition risk estimates with a best-model ROC-AUC of <b>81.0%</b>.",
        s["body"]
    ))
    story.append(Paragraph(
        "The primary findings — that compensation, manager relationship quality, prior job mobility, "
        "and engagement-workload balance are the strongest predictors of attrition — provide specific, "
        "actionable directions for HR retention strategy. The interactive Streamlit dashboard translates "
        "these insights into an accessible interface enabling risk exploration, department-level analysis, "
        "individual employee profiling, and what-if scenario simulation — all grounded in actual model "
        "predictions rather than fabricated values.",
        s["body"]
    ))
    story.append(Paragraph(
        "By moving HR analytics from retrospective reporting to <b>predictive decision intelligence</b>, "
        "this system provides Palo Alto Networks with the capability to identify retention risks earlier, "
        "prioritize interventions more effectively, and measure the potential impact of workforce "
        "investments — while maintaining appropriate human oversight over final employment decisions.",
        s["body"]
    ))

    # ─── 7. RECOMMENDATIONS ───────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("7. Recommendations", s["h1"]))
    section_rule(story)
    recs = [
        ("R1 — Compensation Benchmarking",
         "Conduct an immediate salary equity review for employees in the bottom 25th income percentile "
         "within each job level, prioritizing those with high engagement scores — indicating they are "
         "committed but potentially underpaid."),
        ("R2 — Manager Continuity Programs",
         "Identify employees who recently experienced a manager change and schedule structured check-in "
         "conversations. Prioritize departments with above-average risk indices."),
        ("R3 — Career Mobility Initiative",
         "For employees with YearsSinceLastPromotion ≥ 4 and strong performance ratings, initiate "
         "structured career development conversations and transparent promotion pathways."),
        ("R4 — Overtime Management",
         "For High Risk employees flagged for sustained overtime, evaluate workload sustainability "
         "and implement targeted workload redistribution before attrition materializes."),
        ("R5 — Quarterly Model Review",
         "Retrain the pipeline quarterly as new HR data is collected, and monitor model recall on "
         "fresh test data to detect performance drift over time."),
    ]
    for title, body in recs:
        story.append(finding_box(title, body, s, ACCENT))
        story.append(Spacer(1, 2.5 * mm))

    # ─── 8. REFERENCES ────────────────────────────────────────────────
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("References", s["h1"]))
    section_rule(story)
    refs = [
        "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD '16.",
        "Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Oversampling Technique. JAIR, 16, 321–357.",
        "Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.",
        "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. JMLR, 12, 2825–2830.",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", s["ref"]))

    return story


def add_page_number(canvas, doc):
    """Add footer with page number to every page after cover."""
    if doc.page > 1:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(SLATE)
        canvas.drawString(MARGIN, 1.2 * cm, "Employee Attrition Intelligence — Palo Alto Networks")
        canvas.drawRightString(PAGE_W - MARGIN, 1.2 * cm, f"Page {doc.page - 1}")
        canvas.setStrokeColor(HexColor("#e2e8f0"))
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 1.5 * cm, PAGE_W - MARGIN, 1.5 * cm)

        # Orange accent bar top
        canvas.setFillColor(ORANGE)
        canvas.rect(0, PAGE_H - 0.35 * cm, PAGE_W, 0.35 * cm, fill=1, stroke=0)
        canvas.restoreState()


def main():
    os.makedirs("reports", exist_ok=True)
    doc = SimpleDocTemplate(
        OUT_PATH,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        title="Employee Attrition Prediction — Research Paper",
        author="Sahodev Chandro Ray",
        subject="ML-Based HR Attrition Analytics — Palo Alto Networks",
    )
    story = build_pdf()
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"PDF generated: {OUT_PATH}")


if __name__ == "__main__":
    main()
