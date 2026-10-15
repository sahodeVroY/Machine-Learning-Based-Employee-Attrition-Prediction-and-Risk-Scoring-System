# Machine Learning-Based Employee Attrition Prediction and Risk Scoring System
## Research Paper — Palo Alto Networks HR Analytics

**Author:** Sahodev Chandro Ray · Computer Science and Engineering  
**Date:** October 2026  
**Status:** Completed

---

## Abstract

Employee attrition represents a significant operational and financial challenge for technology organizations. This paper presents the development and evaluation of a machine learning-based attrition prediction and risk scoring system designed for Palo Alto Networks. The system applies a multi-model classification pipeline — comprising Logistic Regression, Random Forest, and XGBoost — to a 1,470-record employee dataset, producing employee-level attrition probability estimates and interpretable risk categories. The best-performing model achieves a ROC-AUC of 0.810, demonstrating meaningful predictive capability beyond chance. Results are surfaced through an interactive Streamlit dashboard enabling HR stakeholders to explore individual employee risk profiles, department-level patterns, and model explanations using SHAP values and feature importance. The system is designed to support proactive, data-driven retention planning rather than reactive workforce management.

---

## 1. Introduction

### 1.1 Background

Employee attrition — the voluntary or involuntary departure of employees from an organization — imposes substantial direct and indirect costs. Tangible costs include recruitment, onboarding, and training; intangible costs include lost institutional knowledge, reduced team cohesion, and productivity disruption. For technology firms like Palo Alto Networks, where specialized talent is critical and competitive labor markets intensify turnover risk, the ability to anticipate and respond to attrition risk proactively is a strategic priority.

Traditional HR analytics has largely been retrospective: reporting on attrition rates after the fact, or identifying patterns in historical exit data. This approach provides limited time for intervention. A predictive system that identifies employees with elevated departure risk in advance enables HR teams to act before attrition occurs — through retention conversations, compensation adjustments, career development support, or targeted engagement programs.

### 1.2 Problem Statement

Palo Alto Networks faces three interconnected attrition challenges:

1. **Unanticipated departures** — Resignations that arrive without warning, leaving insufficient time for knowledge transfer, succession planning, or counter-offers.
2. **Loss of critical talent** — High-performing or specialized employees whose departure is disproportionately disruptive.
3. **Reactive retention** — Counter-offer strategies deployed after an employee has already decided to leave, which are less effective and more costly than proactive intervention.

The proposed system directly addresses these challenges by providing a systematic, quantitative mechanism for estimating individual employee attrition risk in advance.

### 1.3 Objectives

- Develop a machine learning pipeline that classifies employees as likely to attrite or remain.
- Generate calibrated attrition probability estimates for each employee.
- Assign employees to interpretable risk categories (Low, Medium, High).
- Identify the features most strongly associated with predicted attrition.
- Provide an interactive dashboard enabling HR stakeholders to explore findings without requiring technical expertise.

---

## 2. Dataset

### 2.1 Data Source

The dataset contains employee-level records for Palo Alto Networks personnel. It is structured as a tabular dataset with 1,470 employee records and 35 attributes spanning demographic, compensation, satisfaction, career, workload, and organizational dimensions.

### 2.2 Target Variable

The target variable is `Attrition`, a binary indicator:

| Value | Meaning |
|---|---|
| `1` | Employee left the organization |
| `0` | Employee remained |

The dataset exhibits class imbalance: approximately 16.1% of employees attrited (237 records), with 83.9% remaining (1,233 records).

### 2.3 Feature Definitions

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Employee age in years |
| BusinessTravel | Categorical | Travel frequency |
| DailyRate / HourlyRate / MonthlyRate | Numeric | Compensation components |
| Department | Categorical | Organizational department |
| DistanceFromHome | Numeric | Commute distance |
| Education / EducationField | Ordinal / Categorical | Academic background |
| EnvironmentSatisfaction | Ordinal (1–4) | Workplace environment rating |
| Gender | Categorical | Employee gender |
| JobInvolvement | Ordinal (1–4) | Level of job engagement |
| JobLevel | Ordinal (1–5) | Seniority level |
| JobRole | Categorical | Specific designation |
| JobSatisfaction | Ordinal (1–4) | Satisfaction with role |
| MaritalStatus | Categorical | Single / Married / Divorced |
| MonthlyIncome | Numeric | Monthly salary |
| NumCompaniesWorked | Numeric | Prior employer count |
| OverTime | Binary | Overtime worked |
| PercentSalaryHike | Numeric | Last appraisal salary increase |
| PerformanceRating | Ordinal | Performance evaluation |
| RelationshipSatisfaction | Ordinal (1–4) | Workplace relationship quality |
| StockOptionLevel | Ordinal | Stock option grant level |
| TotalWorkingYears | Numeric | Total career experience |
| TrainingTimesLastYear | Numeric | Training sessions attended |
| WorkLifeBalance | Ordinal (1–4) | WLB self-rating |
| YearsAtCompany | Numeric | Organizational tenure |
| YearsInCurrentRole | Numeric | Time in current position |
| YearsSinceLastPromotion | Numeric | Promotion recency |
| YearsWithCurrManager | Numeric | Manager relationship duration |

---

## 3. Methodology

### 3.1 Data Preprocessing

The preprocessing pipeline applied the following steps in sequence:

**A. Data Cleaning**

- Removed constant columns: `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours` (zero variance, non-informative).
- Converted string target (`Yes/No`) to binary integer (1/0).
- Converted `OverTime` from string to binary.
- Removed rows with null target values (none were found in this dataset).

**B. Categorical Encoding**

All remaining categorical string columns were encoded using Label Encoding. This approach was chosen due to the tree-based nature of the primary candidate models, for which label encoding is appropriate and does not impose spurious ordinality.

**C. Feature Scaling**

All numerical features were standardized using `StandardScaler` (zero mean, unit variance). This is required for Logistic Regression and ensures consistent feature scales across all models.

**D. Train-Test Split**

The dataset was divided 80% / 20% using a stratified split to preserve the attrition class ratio in both subsets:

- Training set: 1,176 records
- Test set: 294 records

**E. Class Imbalance Handling**

SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set after splitting to avoid data leakage. SMOTE generates synthetic minority-class samples by interpolating between existing minority instances, increasing the representation of attrition cases in training without replicating exact observations.

Post-SMOTE training set size: 1,976 balanced records (988 per class).

### 3.2 Feature Engineering

Four composite features were constructed to capture relationships not expressed by individual raw variables:

| Feature | Formula / Logic | Rationale |
|---|---|---|
| `IncomeExperienceRatio` | `MonthlyIncome / (TotalWorkingYears + 1)` | Captures compensation-relative-to-experience; underpaid experienced employees may be at higher risk |
| `PromotionDelayFlag` | `1 if YearsSinceLastPromotion ≥ 4` | Flags prolonged career stagnation |
| `EngagementScore` | Mean of JobSatisfaction, EnvironmentSatisfaction, RelationshipSatisfaction, WorkLifeBalance, JobInvolvement | Composite engagement indicator |
| `WorkloadStressFlag` | Binary: 1 if OverTime = Yes | Explicit workload pressure indicator |

Feature engineering was applied before the train-test split to ensure consistent feature sets, with no leakage risk given these features depend only on raw input variables available at prediction time.

### 3.3 Model Development

Three classification algorithms were trained and evaluated:

#### Logistic Regression
A linear classifier providing an interpretable baseline. Trained with `class_weight='balanced'` to counteract class imbalance, and `max_iter=1000` for convergence on scaled features. Coefficients provide directional insight into feature-attrition relationships.

#### Random Forest
A bagged ensemble of 300 decision trees. Trained with `class_weight='balanced'`, `min_samples_leaf=2` (regularization), and `n_estimators=300`. Captures non-linear relationships and feature interactions. Provides Gini-impurity-based feature importance.

#### XGBoost
A gradient-boosted tree ensemble with `n_estimators=400`, `learning_rate=0.05`, `max_depth=6`, and `scale_pos_weight` set to the inverse class ratio to handle imbalance. XGBoost provides gain-based feature importance and is well-suited to tabular classification problems.

### 3.4 Risk Scoring Framework

Attrition probabilities from the selected model are mapped to three risk categories using configurable thresholds:

```
Attrition Probability (p)
    │
    ├── p < 0.30  →  🟢 Low Risk
    ├── 0.30 ≤ p ≤ 0.60  →  🟡 Medium Risk
    └── p > 0.60  →  🔴 High Risk
```

Thresholds are adjustable through the dashboard interface, allowing HR teams to tune sensitivity based on organizational priorities and available retention resources.

### 3.5 Explainability

Two complementary explainability approaches are implemented:

**Global Feature Importance**: For tree-based models, mean decrease in impurity (Gini importance) is used. For Logistic Regression, absolute coefficient magnitudes provide directional importance. Both are visualized as ranked horizontal bar charts.

**SHAP (SHapley Additive exPlanations)**: SHAP values decompose each model prediction into additive contributions from individual features. For tree models, `TreeExplainer` is used (exact, fast). A `KernelExplainer` fallback ensures compatibility with all model types. SHAP enables per-employee explanations showing which specific attributes are increasing or decreasing that employee's predicted attrition risk.

---

## 4. Results

### 4.1 Model Evaluation

All three models were evaluated on the held-out test set (294 records) using five classification metrics:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 73.8% | 35.0% | **74.5%** | **47.6%** | **81.0%** |
| XGBoost | **87.1%** | **69.6%** | 34.0% | 45.7% | 80.7% |
| Random Forest | **87.1%** | 65.5% | 40.4% | 45.7% | 78.9% |

**Model Selection Rationale:** Logistic Regression was selected as the final model based on its superior ROC-AUC (81.0%) and highest Recall (74.5%). In attrition prediction, **recall is the most critical operational metric**: missing an at-risk employee (false negative) is typically more costly than an unnecessary retention conversation (false positive). Logistic Regression's higher recall means it correctly identifies approximately 3 in 4 employees who will actually leave, compared to only 1 in 3 for the tree-based models on the test set, despite their higher accuracy. The ROC-AUC of 81.0% indicates strong discrimination between attrition and retention cases across all probability thresholds.

### 4.2 Risk Distribution

Applying the selected model to all 1,470 employees:

| Risk Category | Count | Percentage |
|---|---|---|
| 🟢 Low Risk | 742 | 50.5% |
| 🟡 Medium Risk | 377 | 25.6% |
| 🔴 High Risk | 351 | 23.9% |

The model identifies 351 employees (23.9%) as high-risk — a manageable prioritization cohort for targeted HR intervention.

### 4.3 Key Attrition Predictors

Feature importance from the trained Logistic Regression model (absolute coefficient magnitude):

| Rank | Feature | Importance |
|---|---|---|
| 1 | MonthlyIncome | 0.7462 |
| 2 | YearsWithCurrManager | 0.5417 |
| 3 | NumCompaniesWorked | 0.5250 |
| 4 | YearsAtCompany | 0.5008 |
| 5 | Department | 0.4756 |
| 6 | EngagementScore | 0.4044 |
| 7 | WorkloadStressFlag / OverTime | 0.3902 |
| 8 | MaritalStatus | 0.3868 |
| 9 | YearsInCurrentRole | 0.3806 |
| 10 | IncomeExperienceRatio | 0.3640 |

### 4.4 Key Findings

**Finding 1 — Compensation is the Dominant Predictor**  
`MonthlyIncome` is the single most influential predictor (importance: 0.746). Employees with lower monthly income relative to their cohort exhibit substantially elevated predicted attrition risk, suggesting compensation competitiveness as the primary lever for retention intervention.

**Finding 2 — Manager Relationship Quality Matters**  
`YearsWithCurrManager` ranks second (importance: 0.542). Both very short tenure with the current manager (potential disruption from a recent manager change) and unusually long tenure without corresponding role advancement could contribute to elevated risk.

**Finding 3 — Job Mobility History Predicts Future Mobility**  
`NumCompaniesWorked` ranks third (0.525), confirming that employees with a pattern of changing employers are statistically more likely to do so again. This reinforces the value of behavioral hiring criteria aligned with retention goals.

**Finding 4 — Tenure and Stagnation**  
`YearsAtCompany` (0.501) and `YearsInCurrentRole` (0.381) together indicate that both very early tenure (adjustment risk) and extended tenure without role advancement (stagnation risk) contribute to elevated attrition probability.

**Finding 5 — Overtime and Engagement are Compounding Factors**  
The engineered `WorkloadStressFlag` / `OverTime` (0.390) and `EngagementScore` (0.404) both appear in the top 10, suggesting that employees experiencing both high workload and low satisfaction represent a compounding risk profile requiring priority intervention.

---

## 5. Discussion

### 5.1 Practical Implications for HR

The system's output enables HR teams to move from broad workforce statistics to targeted, evidence-based retention strategies:

- **High-Risk employees (n=351)**: Priority candidates for immediate retention conversations, compensation reviews, and engagement check-ins — particularly those flagged for low income relative to peers, low engagement scores, and overtime patterns.
- **Department-level patterns**: The heatmap visualization identifies which departments and job roles have disproportionate concentrations of high-risk employees, enabling resource allocation to where retention pressure is greatest.
- **What-If Scenario Analysis**: The dashboard's scenario simulator allows HR partners to model the expected change in a specific employee's attrition probability if, for example, overtime is eliminated or a salary adjustment is made — providing a quantified basis for retention ROI estimates.

### 5.2 Limitations

- **Causal inference**: Feature importance does not establish causation. Monthly income being the top predictor does not prove salary is the single cause of attrition; unmeasured confounders (manager quality, team dynamics, external offers) may also play a role.
- **Temporal validity**: The model is trained on a snapshot dataset. Workforce conditions, compensation markets, and organizational culture evolve over time, and the model should be periodically re-validated and retrained.
- **Self-report bias**: Satisfaction and engagement variables are self-reported ratings subject to social desirability bias and assessment timing effects.
- **Class threshold sensitivity**: The choice of 30% and 60% risk thresholds are defaults; organizations should calibrate these based on their intervention capacity and risk tolerance.

### 5.3 Responsible Use Considerations

This system is designed as a **decision-support tool**, not an automated personnel decision system. Model predictions:
- Should be reviewed by qualified HR professionals with contextual knowledge of the employee.
- Must not be used as the sole basis for employment-related decisions.
- Should be audited for differential performance across protected demographic groups.
- Are subject to data minimization and employee privacy requirements.

---

## 6. Conclusion

This paper presented a machine learning-based employee attrition prediction and risk scoring system for Palo Alto Networks. Through a structured pipeline encompassing data preprocessing, feature engineering, multi-model training, and SHAP-based explainability, the system produces actionable, employee-level attrition risk estimates with a best-model ROC-AUC of 81.0%.

The primary findings — that compensation, manager relationship quality, prior job mobility, and engagement-workload balance are the strongest predictors of attrition — provide specific, actionable directions for HR retention strategy. The interactive Streamlit dashboard translates these insights into an accessible interface for HR stakeholders, enabling risk exploration, department-level analysis, individual employee profiling, and what-if scenario simulation.

By moving HR analytics from retrospective reporting to predictive decision intelligence, this system provides Palo Alto Networks with the capability to identify retention risks earlier, prioritize interventions more effectively, and measure the potential impact of workforce investments — all while maintaining appropriate human oversight over final employment decisions.

---

## 7. Recommendations

Based on the analysis, the following retention interventions are recommended for immediate consideration:

1. **Compensation Benchmarking**: Conduct an immediate salary equity review for employees in the bottom 25th income percentile within each job level, with particular focus on employees who also have High EngagementScore — indicating they are engaged but potentially underpaid.

2. **Manager Continuity Programs**: Identify employees who recently experienced a manager change and schedule structured check-in conversations. Prioritize departments where the Department Risk Index is above average.

3. **Career Mobility Initiative**: For employees with `YearsSinceLastPromotion ≥ 4` and strong performance ratings, initiate structured career development conversations and transparent promotion pathways.

4. **Overtime Management**: For High Risk employees who are also flagged for overtime, evaluate workload sustainability and implement targeted workload redistribution before attrition materializes.

5. **Quarterly Model Review**: Retrain the pipeline quarterly as new HR data is collected, and monitor model recall on fresh test data to detect performance drift over time.

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Oversampling Technique. *JAIR*, 16, 321–357.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.

---

*This document was produced as part of the Palo Alto Networks Employee Attrition Intelligence project.*  
*Model artifacts, source code, and interactive dashboard are available in the project repository.*
