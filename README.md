# 🔥 Employee Attrition Intelligence
### ML-Powered HR Risk Analytics Platform — Palo Alto Networks

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-yellow)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Predict employee attrition risk. Explain it. Act on it.**
> 
> A production-ready, ML-driven HR analytics system that identifies employees at elevated attrition risk, assigns interpretable risk scores, and surfaces the key factors driving predictions — all through an interactive Streamlit dashboard.

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📐 System Architecture

```
Employee Dataset (1,470 records)
        │
        ▼
Data Preprocessing ── Encoding · Scaling · SMOTE
        │
        ▼
Feature Engineering ── Income-Exp Ratio · Promotion Delay
        │              Engagement Score · Workload Flag
        ▼
Model Training ─────── Logistic Regression
        │               Random Forest
        │               XGBoost
        ▼
Model Evaluation ────── Accuracy · Precision · Recall · F1 · ROC-AUC
        │
        ▼
Risk Scoring ──────────  < 30%  → 🟢 Low Risk
        │                30–60% → 🟡 Medium Risk
        │                > 60%  → 🔴 High Risk
        ▼
Streamlit Dashboard ─── Risk Overview · Employee Profiles
                         Dept Analysis · SHAP Explainability
                         What-If Simulator · Model Performance
```

---

## 🗂️ Project Structure

```
Project-2/
├── data/
│   └── raw/
│       └── employee_data.csv         ← Palo Alto Networks dataset
├── src/
│   ├── data_preprocessing.py         ← Clean, encode, scale, SMOTE
│   ├── feature_engineering.py        ← 4 engineered features
│   ├── train_models.py               ← LR, RF, XGBoost training
│   ├── evaluate_models.py            ← 5-metric evaluation
│   ├── risk_scoring.py               ← Probability → risk category
│   └── explainability.py             ← Feature importance + SHAP
├── app/
│   └── app.py                        ← Streamlit dashboard (6 pages)
├── models/                           ← Saved .pkl artifacts (git-ignored large files)
├── .streamlit/
│   └── config.toml                   ← Theme + server settings
├── run_pipeline.py                   ← End-to-end training pipeline
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/employee-attrition-intelligence.git
cd employee-attrition-intelligence
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the models (one-time setup)
```bash
python run_pipeline.py
```

### 4. Launch the dashboard
```bash
streamlit run app/app.py
```
Opens at **http://localhost:8501**

---

## ☁️ Streamlit Cloud Deployment

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set:
   - **Repository**: `your-username/employee-attrition-intelligence`
   - **Branch**: `main`
   - **Main file**: `app/app.py`
4. Click **Deploy**

> **Important:** The `models/` directory must contain the trained `.pkl` files.
> Since large RF/XGBoost models are gitignored, either:
> - Use **Git LFS** to track them, OR
> - Add a startup script that runs `run_pipeline.py` on first deploy (see below)

**Auto-train on deploy** — add to `app/app.py` top-level:
```python
import os, subprocess
if not os.path.exists("models/best_model_name.pkl"):
    subprocess.run(["python", "run_pipeline.py"], check=True)
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 73.8% | 35.0% | 74.5% | 47.6% | **81.0%** |
| XGBoost | 87.1% | 69.6% | 34.0% | 45.7% | 80.7% |
| Random Forest | 87.1% | 65.5% | 40.4% | 45.7% | 78.9% |

> Logistic Regression selected as best model by ROC-AUC, with superior recall for attrition cases — critical for HR use cases where missing a high-risk employee is costlier than a false alarm.

---

## 🔍 Top Attrition Predictors

| Rank | Feature | Description |
|---|---|---|
| 1 | MonthlyIncome | Lower compensation linked to higher attrition risk |
| 2 | YearsWithCurrManager | Manager relationship stability |
| 3 | NumCompaniesWorked | Prior job-hopping behavior |
| 4 | YearsAtCompany | Organizational tenure |
| 5 | EngagementScore | Composite job + env + WLB satisfaction |

---

## 🎛️ Dashboard Features

| Page | Features |
|---|---|
| 📊 Risk Overview | KPI cards, risk donut chart, probability histogram, dept/OT breakdowns |
| 👤 Employee Profiles | Searchable table, per-employee gauge, SHAP bar chart, CSV export |
| 🏢 Department Analysis | Dept risk KPIs, role heatmap, high-risk role ranking |
| 🔍 Explainability | Global feature importance, SHAP summary (sample of 200) |
| 🎲 What-If Simulator | Live attribute sliders, before/after gauges, risk delta |
| 📈 Model Performance | Multi-model metric comparison, active model selector |

Sidebar controls: **Department · Job Role · Age Group · Risk Level · Threshold Sliders · Model Selector**

---

## 🛡️ Responsible Use

Risk scores are **decision-support signals**, not autonomous employment decisions. Model predictions should always be:
- Reviewed by qualified HR professionals
- Combined with qualitative employee context
- Monitored for bias across demographic groups
- Re-validated periodically as workforce conditions change

---

## 👤 Author

**Sahodev Chandro Ray**  
Computer Science and Engineering  

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
