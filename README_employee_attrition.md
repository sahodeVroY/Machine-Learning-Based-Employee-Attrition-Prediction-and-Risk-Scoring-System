# Machine Learning-Based Employee Attrition Prediction and Risk Scoring System

## Overview

The **Machine Learning-Based Employee Attrition Prediction and Risk
Scoring System** is a predictive HR analytics solution designed to help
organizations identify employees who may be at elevated risk of leaving
and understand the factors associated with that risk.

The system moves HR analytics beyond descriptive and diagnostic
reporting toward **predictive decision intelligence** by combining
machine learning classification, employee-level risk scoring, model
explainability, and an interactive Streamlit dashboard.

### Core Objectives

-   Identify employees who are at elevated risk of attrition.
-   Estimate an individual employee's probability of leaving.
-   Categorize employees into low-, medium-, and high-risk groups.
-   Identify important factors associated with attrition predictions.
-   Support targeted and proactive retention interventions.
-   Enable data-driven workforce planning.

------------------------------------------------------------------------

## Background and Context

HR leaders often need to answer a critical question:

> **Which employees are most likely to leave the organization in the
> near future?**

Traditional HR reporting can describe historical workforce patterns, but
it may not provide a systematic mechanism for estimating future
attrition risk at the employee level.

This project introduces predictive intelligence to support:

-   **Early identification of high-risk employees**
-   **Targeted retention interventions**
-   **Data-driven workforce planning**

------------------------------------------------------------------------

## Problem Statement

Palo Alto Networks faces employee-retention challenges such as:

-   Sudden, unanticipated resignations.
-   Loss of high-performing or critical employees.
-   Reactive counter-offers that may occur too late.

The organization requires a more systematic approach to:

-   Predict employee attrition risk.
-   Generate quantitative risk scores for individual employees.
-   Understand the key factors influencing employee exit decisions.

The proposed system addresses these needs through machine learning-based
attrition prediction and explainable employee-level risk scoring.

------------------------------------------------------------------------

## Solution Overview

The system follows an end-to-end predictive analytics pipeline:

``` text
Employee Dataset
      │
      ▼
Data Preprocessing
      │
      ├── Categorical Encoding
      ├── Numerical Scaling
      └── Class Imbalance Handling
      │
      ▼
Feature Engineering
      │
      ▼
Model Development
      │
      ├── Logistic Regression
      ├── Random Forest
      └── Gradient Boosting / XGBoost
      │
      ▼
Model Evaluation
      │
      ├── Accuracy
      ├── Precision
      ├── Recall
      ├── F1-Score
      └── ROC-AUC
      │
      ▼
Attrition Probability
      │
      ▼
Risk Scoring
      │
      ├── Low Risk
      ├── Medium Risk
      └── High Risk
      │
      ▼
Model Explainability
      │
      ├── Feature Importance
      └── SHAP Explanations
      │
      ▼
Streamlit HR Dashboard
```

------------------------------------------------------------------------

## Key Features

### 1. Employee Attrition Prediction

The system predicts whether an employee is likely to leave the
organization using historical employee and workplace attributes.

The primary target variable is:

``` text
Attrition
```

where:

-   `0` = Employee did not leave.
-   `1` = Employee left.

### 2. Employee-Level Risk Scoring

Each employee receives an estimated **attrition probability between 0
and 1**.

The probability is then converted into an interpretable risk category.

  Risk Category       Attrition Probability
  ----------------- -----------------------
  **Low Risk**                      `< 30%`
  **Medium Risk**                  `30–60%`
  **High Risk**                     `> 60%`

These thresholds can also be adjusted interactively through the
Streamlit dashboard.

### 3. Explainable Predictions

The system is designed to provide insight into **why** an employee
receives a particular risk score.

Possible explanations include:

-   Feature importance.
-   SHAP value explanations.
-   Individual-level contributing factors.

For example, an employee's elevated predicted risk may be associated
with a combination such as:

``` text
Low Job Satisfaction
        +
High Overtime
        +
Long Time Without Promotion
        ↓
Elevated Predicted Attrition Risk
```

These explanations are intended to support informed HR analysis rather
than treating model predictions as standalone decisions.

### 4. Department-Level Risk Analysis

The dashboard aggregates risk information across:

-   Departments.
-   Job roles.

This enables HR users to identify areas with higher concentrations of
predicted attrition risk.

### 5. What-If Scenario Exploration

The explainability interface includes support for exploring how changes
to relevant employee attributes could affect the model's prediction.

------------------------------------------------------------------------

# Dataset

The project uses employee-level HR data containing demographic,
compensation, satisfaction, career, workload, and organizational
variables.

## Dataset Fields

  -----------------------------------------------------------------------
  Field                               Description
  ----------------------------------- -----------------------------------
  `Age`                               Age of the employee in years

  `Attrition`                         Indicates whether the employee left
                                      the organization (`0 / 1`)

  `BusinessTravel`                    Frequency of business travel
                                      (`Non-Travel`, `Travel_Rarely`,
                                      `Travel_Frequently`)

  `DailyRate`                         Daily compensation rate of the
                                      employee

  `Department`                        Department where the employee
                                      works, such as R&D, Sales, or HR

  `DistanceFromHome`                  Distance between the employee's
                                      home and workplace

  `Education`                         Education level

  `EducationField`                    Field of education, such as Life
                                      Sciences, Medical, or Technical

  `EnvironmentSatisfaction`           Satisfaction with the work
                                      environment (`1–4`)

  `Gender`                            Employee gender

  `HourlyRate`                        Hourly wage of the employee

  `JobInvolvement`                    Level of involvement in work
                                      (`1–4`)

  `JobLevel`                          Job seniority level (`1–5`)

  `JobRole`                           Specific job role or designation

  `JobSatisfaction`                   Satisfaction with the job role
                                      (`1–4`)

  `MaritalStatus`                     Marital status (`Single`,
                                      `Married`, `Divorced`)

  `MonthlyIncome`                     Monthly salary earned by the
                                      employee

  `MonthlyRate`                       Monthly compensation rate

  `NumCompaniesWorked`                Number of companies the employee
                                      has previously worked for

  `OverTime`                          Indicates whether the employee
                                      works overtime (`Yes / No`)

  `PercentSalaryHike`                 Percentage increase in salary
                                      during the last appraisal

  `PerformanceRating`                 Employee performance evaluation
                                      score

  `RelationshipSatisfaction`          Satisfaction with workplace
                                      relationships (`1–4`)

  `StockOptionLevel`                  Level of stock options granted to
                                      the employee

  `TotalWorkingYears`                 Total professional experience
                                      across all companies

  `TrainingTimesLastYear`             Number of training programs
                                      attended during the last year

  `WorkLifeBalance`                   Work-life balance rating (`1–4`)

  `YearsAtCompany`                    Number of years the employee has
                                      worked at the company

  `YearsInCurrentRole`                Number of years in the current role

  `YearsSinceLastPromotion`           Number of years since the last
                                      promotion

  `YearsWithCurrManager`              Number of years working with the
                                      current manager
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# Data Science Methodology

## Step 1 --- Data Preprocessing

The preprocessing pipeline includes:

### Categorical Encoding

Categorical variables can be transformed using:

-   Label Encoding.
-   One-Hot Encoding.

### Numerical Feature Scaling

Numerical variables are scaled where required by the selected modeling
approach.

### Class Imbalance Handling

Because employee attrition datasets may contain fewer attrition cases
than non-attrition cases, the project supports:

-   **SMOTE (Synthetic Minority Oversampling Technique)**.
-   **Class weights**.

The selected approach should be determined based on the training-data
distribution and model performance.

### Train-Test Split

The dataset is divided into training and testing subsets for model
development and unbiased evaluation.

A stratified split should be preferred when appropriate so that the
attrition class distribution is represented consistently across subsets.

------------------------------------------------------------------------

## Step 2 --- Feature Engineering

The system creates additional features intended to capture relationships
that may not be directly represented by individual raw variables.

Planned engineered features include:

### Income-to-Experience Ratio

A relationship between compensation and professional experience.

### Promotion Delay Indicator

A feature representing extended time since the employee's most recent
promotion.

### Engagement Composite Score

A combined indicator derived from relevant engagement and satisfaction
variables.

### Workload Stress Flag

A derived indicator intended to capture potentially elevated workload
pressure, including overtime-related patterns.

Feature engineering should be validated to avoid leakage from
information that would not be available at prediction time.

------------------------------------------------------------------------

# Step 3 --- Model Development

The project uses a baseline model together with more advanced tree-based
classifiers.

## Logistic Regression

**Logistic Regression** serves as the interpretability-oriented baseline
model.

Advantages:

-   Simple.
-   Fast to train.
-   Provides a strong baseline for binary classification.
-   Coefficients can help explain directional relationships.

## Random Forest Classifier

Random Forest provides a non-linear ensemble approach capable of
modeling interactions among employee attributes.

Potential advantages:

-   Handles non-linear relationships.
-   Captures feature interactions.
-   Provides feature importance estimates.
-   Generally requires less feature transformation than linear models.

## Gradient Boosting / XGBoost

Gradient boosting models provide an additional advanced classification
approach for identifying complex patterns in employee attrition data.

Potential implementation:

``` text
XGBoost
```

or another suitable gradient-boosting classifier.

------------------------------------------------------------------------

# Step 4 --- Model Evaluation

Models are evaluated using multiple classification metrics rather than
accuracy alone.

  Metric          Purpose
  --------------- ------------------------------------------------
  **Accuracy**    Overall classification correctness
  **Precision**   Controls false-positive predictions
  **Recall**      Measures attrition detection capability
  **F1-Score**    Balances precision and recall
  **ROC-AUC**     Measures overall classification discrimination

### Why Recall Matters

For an attrition-risk system, missing employees who are actually at risk
can be particularly important. Therefore, recall should be examined
alongside precision rather than relying solely on overall accuracy.

### Why F1-Score Matters

F1-score provides a single measure that balances precision and recall,
which is useful when the target classes are imbalanced.

------------------------------------------------------------------------

# Step 5 --- Risk Scoring Framework

The model produces an employee-level probability:

``` text
Attrition Probability ∈ [0, 1]
```

This probability is mapped into three risk categories:

``` text
Probability
    │
    ├── < 30% ─────────► Low Risk
    │
    ├── 30–60% ────────► Medium Risk
    │
    └── > 60% ────────► High Risk
```

## Risk Categories

### Low Risk

``` text
Attrition Probability < 30%
```

Represents a lower predicted probability of attrition according to the
model.

### Medium Risk

``` text
30% ≤ Attrition Probability ≤ 60%
```

Represents an intermediate predicted probability of attrition.

### High Risk

``` text
Attrition Probability > 60%
```

Represents a higher predicted probability of attrition.

> Risk categories are model outputs and should be treated as
> decision-support signals rather than definitive statements about an
> employee's future behavior.

------------------------------------------------------------------------

# Step 6 --- Model Explainability

Explainability is an important part of the system because HR users need
to understand the factors associated with model predictions.

## Feature Importance

Global feature importance can be used to identify which variables
contribute most strongly to the model's predictions.

## SHAP

**SHAP (SHapley Additive exPlanations)** can optionally be used for:

-   Global model interpretation.
-   Individual prediction explanations.
-   Understanding positive and negative feature contributions.

## Individual-Level Reasoning

The system can surface combinations of attributes that contribute to an
individual prediction.

For example:

``` text
Employee
   │
   ├── Job Satisfaction
   ├── OverTime
   ├── Years Since Last Promotion
   └── Work-Life Balance
          │
          ▼
   Model Prediction
          │
          ▼
   Attrition Probability
          │
          ▼
   Risk Category
```

------------------------------------------------------------------------

# Streamlit Web Application

The project includes an interactive **Streamlit** dashboard for HR
analytics and model-driven risk exploration.

## Dashboard Modules

### 1. Attrition Risk Dashboard

Displays:

-   Overall risk distribution.
-   High-risk employee counts.

Potential visualizations include:

-   Risk-category distribution charts.
-   High-risk employee KPI cards.
-   Attrition probability summaries.

### 2. Employee Risk Profile

Provides an individual employee view containing:

-   Employee-level attrition probability.
-   Risk category.
-   Key contributing factors.

### 3. Department-Level Risk View

Provides aggregated risk analysis by:

-   Department.
-   Job role.

This allows users to compare predicted risk patterns across
organizational groups.

### 4. Explainability Panel

Provides:

-   Feature importance.
-   Individual prediction explanations.
-   What-if scenario exploration.

------------------------------------------------------------------------

# User Capabilities

The dashboard should allow users to:

### Department & Role Filters

Filter the analysis by:

``` text
Department
Job Role
```

### Risk Threshold Sliders

Adjust the thresholds used to define risk categories or focus the
dashboard on selected risk levels.

### Employee ID Selector

Select an individual employee to inspect their:

-   Predicted attrition probability.
-   Risk category.
-   Contributing factors.
-   Model explanation.

------------------------------------------------------------------------

# Recommended Dashboard Layout

A possible Streamlit layout is:

``` text
┌───────────────────────────────────────────────────────┐
│        Employee Attrition Risk Dashboard              │
├───────────────────────────────────────────────────────┤
│ Department │ Role │ Risk Threshold │ Employee ID      │
├───────────────────────────────────────────────────────┤
│ Total Employees │ High Risk │ Avg. Risk │ Attrition  │
├───────────────────────────────────────────────────────┤
│                                                       │
│          Overall Risk Distribution                    │
│                                                       │
├───────────────────────────┬───────────────────────────┤
│ Employee Risk Profile     │ Feature Importance        │
│                           │                           │
│ Probability: XX%          │ Feature A                │
│ Risk: High                │ Feature B                │
│                           │ Feature C                │
├───────────────────────────┴───────────────────────────┤
│ Department / Role Risk Analysis                        │
├───────────────────────────────────────────────────────┤
│ What-If Scenario / Explainability                     │
└───────────────────────────────────────────────────────┘
```

------------------------------------------------------------------------

# Technology Stack

The implementation can be built using the following Python ecosystem:

-   **Python** --- Core programming language.
-   **Pandas** --- Data manipulation and analysis.
-   **NumPy** --- Numerical computation.
-   **Scikit-learn** --- Preprocessing, Logistic Regression, Random
    Forest, evaluation, and related ML utilities.
-   **XGBoost** --- Gradient boosting classification.
-   **imbalanced-learn** --- SMOTE and class-imbalance handling.
-   **SHAP** --- Optional model explainability.
-   **Streamlit** --- Interactive web application.
-   **Matplotlib / Seaborn / Plotly** --- Data visualization.

> Exact package versions should be pinned in `requirements.txt` after
> the implementation environment has been finalized.

------------------------------------------------------------------------

# Suggested Project Structure

``` text
employee-attrition-prediction/
│
├── data/
│   ├── raw/
│   │   └── employee_data.csv
│   └── processed/
│       └── processed_employee_data.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_explainability.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── risk_scoring.py
│   └── explainability.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── gradient_boosting.pkl
│
├── app/
│   └── app.py
│
├── reports/
│   ├── research_paper/
│   └── executive_summary/
│
├── requirements.txt
├── README.md
└── .gitignore
```

------------------------------------------------------------------------

# Installation

## 1. Clone the Repository

``` bash
git clone <repository-url>
cd employee-attrition-prediction
```

## 2. Create a Virtual Environment

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Running the Streamlit Application

From the project root:

``` bash
streamlit run app/app.py
```

The application will normally be available at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# Typical Machine Learning Workflow

``` text
1. Load employee dataset
        ↓
2. Inspect data and perform EDA
        ↓
3. Clean and preprocess data
        ↓
4. Encode categorical variables
        ↓
5. Scale numerical features
        ↓
6. Handle class imbalance
        ↓
7. Engineer predictive features
        ↓
8. Split into train/test sets
        ↓
9. Train baseline model
        ↓
10. Train advanced models
        ↓
11. Evaluate models
        ↓
12. Select the appropriate model
        ↓
13. Generate attrition probabilities
        ↓
14. Convert probabilities into risk categories
        ↓
15. Generate explainability outputs
        ↓
16. Display results in Streamlit
```

------------------------------------------------------------------------

# Example Prediction Workflow

A single employee can be processed through the system as follows:

``` text
Employee Attributes
       │
       ▼
Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Trained ML Model
       │
       ▼
Attrition Probability
       │
       ├── 0.18 → Low Risk
       ├── 0.47 → Medium Risk
       └── 0.78 → High Risk
       │
       ▼
Explanation
       │
       ├── Important features
       ├── Feature contributions
       └── Individual-level factors
```

------------------------------------------------------------------------

# Data Science and Business Interpretation

The system is designed to support HR decision-making rather than replace
human judgment.

A high predicted attrition probability should be interpreted as:

> **The model identifies a higher estimated probability of attrition
> based on the information available to it.**

It should not automatically be interpreted as confirmation that an
employee will leave.

Likewise, model explanations describe factors associated with the
prediction; they do not establish that a particular factor is a causal
reason for an employee's behavior.

------------------------------------------------------------------------

# Responsible Use Considerations

Employee attrition prediction involves sensitive workforce information.
A production implementation should therefore consider:

-   Appropriate access controls.
-   Protection of employee-level information.
-   Secure storage of datasets and model artifacts.
-   Monitoring for model bias and disparate performance.
-   Periodic model validation and retraining.
-   Human review before employment-related actions.
-   Clear documentation of how risk scores are generated.
-   Avoiding automatic adverse decisions based solely on model
    predictions.

Risk scores should function as **decision-support signals**, not
automatic determinants of employment outcomes.

------------------------------------------------------------------------

# Expected Business Value

The system is designed to support:

-   Earlier identification of employees with elevated predicted
    attrition risk.
-   More targeted retention planning.
-   Better visibility into workforce risk patterns.
-   Department- and role-level risk analysis.
-   More transparent ML-assisted HR analytics.
-   Data-driven workforce planning.
-   Proactive rather than purely reactive retention analysis.

------------------------------------------------------------------------

# Deliverables

The project requires three primary deliverables.

## 1. Research Paper

The research paper should contain:

-   Exploratory Data Analysis (EDA).
-   Data science methodology.
-   Model development.
-   Model evaluation.
-   Key insights.
-   Attrition-risk findings.
-   Recommendations.

## 2. Streamlit Dashboard

The dashboard provides:

Live HR Analytics: Interactive KPI cards and charts displaying workforce statistics and attrition trends.

Overall Attrition-Risk Distribution: Dynamic visualizations showing low-, medium-, and high-risk employee distributions, updated automatically based on selected filters.

Employee-Level Risk Profiles: Searchable and filterable employee records with attrition probabilities, risk categories, and individual risk-factor explanations.

Department and Role Analysis: Interactive comparisons of departments and job roles using attrition rates, employee counts, and average risk scores.

Model Explainability: Feature importance visualizations and individual prediction explanations using SHAP or the project's existing explainability methods.

What-If Scenario Exploration: Interactive sliders and input controls to modify relevant employee attributes, recalculate predictions using the trained model, and compare risk scores before and after changes.

Interactive Filtering and Drill-Down: Global filters for department, job role, age group, and risk level, with synchronized updates across relevant charts, tables, and KPI cards.

Dynamic Visualizations: Interactive Plotly charts with hover details, zooming, selection, and customizable metrics.

Data Export: Download filtered employee records and relevant analytical results as CSV files.

Modern UI/UX: Professional KPI cards, intuitive sidebar navigation, consistent risk color coding, responsive layouts, and clear visual hierarchy.

Implementation Requirements:

Preserve the existing dataset, trained model, preprocessing pipeline, and prediction logic.

Ensure all interactive components work with actual project data rather than fabricated values.

Use Streamlit session state and caching where appropriate.

Validate inputs and handle missing data and prediction errors gracefully.

Ensure that filters, charts, buttons, and scenario controls update the dashboard correctly.

Test all dashboard features and confirm that the application runs without errors.

## 3. Executive Summary

A concise summary intended for **government stakeholders**, presenting:

-   Major findings.
-   Key insights.
-   Risk patterns.
-   Recommendations.
-   Potential organizational implications.

------------------------------------------------------------------------

# Future Enhancements

Potential future improvements include:

-   Automated model retraining.
-   Real-time HR data integration.
-   Advanced calibration of predicted probabilities.
-   SHAP-based interactive explanations.
-   Fairness and bias monitoring.
-   Employee risk trend tracking over time.
-   Automated HR reporting.
-   Model monitoring and drift detection.
-   Role- and department-specific models.
-   More advanced what-if simulations.
-   Secure cloud deployment.

------------------------------------------------------------------------

# Conclusion

The **Machine Learning-Based Employee Attrition Prediction and Risk
Scoring System** transforms HR analytics at Palo Alto Networks from
primarily descriptive and diagnostic analysis toward **predictive
decision intelligence**.

By forecasting employee attrition risk, assigning interpretable
employee-level risk scores, and exposing the factors associated with
model predictions, the system provides a structured framework for
proactive workforce analysis.

The combination of machine learning, risk scoring, explainability, and
interactive Streamlit analytics enables HR stakeholders to move from:

``` text
Reactive Retention
        ↓
Historical Reporting
        ↓
Predictive Risk Detection
        ↓
Explainable Workforce Intelligence
        ↓
Proactive Workforce Planning
```

The goal is to provide actionable analytical insight while keeping final
workforce decisions subject to appropriate human judgment and
organizational policies.

------------------------------------------------------------------------

# Project Status

**Status:** In Development

The project requirements, dataset schema, analytical methodology,
risk-scoring framework, explainability requirements, and Streamlit
dashboard specifications have been defined. Implementation, model
training, validation, and deployment remain dependent on the final
development environment and dataset.

------------------------------------------------------------------------

# License

Add the applicable project license here.

For example:

``` text
MIT License
```

if the project is intended to be released under the MIT License.

------------------------------------------------------------------------

# Author

**Sahodev Chandro Ray**

Computer Science and Engineering
