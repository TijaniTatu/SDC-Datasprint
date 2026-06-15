# DataSprint 2026 — Predicting Financial Status in Kenya

Predicting whether a Kenyan adult's financial situation has **Improved**, **Stayed the same**, or **Worsened**, using the 2024 FinAccess Household Survey. Built for the Strathmore Data Community × iLab Africa DataSprint 2026.

## Problem

Despite a decade of progress in financial inclusion, the 2024 FinAccess Household Survey found that **9.9%** of Kenyan adults remain fully excluded from financial services and over half (**52.6%**) report their financial situation has worsened year-on-year. This project builds a multiclass classifier to predict financial status and, more importantly, to identify *which factors drive financial deterioration* and what policymakers, banks, and NGOs should prioritise.

This is a **multiclass classification** problem evaluated on **weighted F1-score** (the target is imbalanced: 52.6% Worsened / 26.9% Stayed the same / 20.5% Improved).

## Data

- **Source:** [2024 FinAccess Household Survey (Kaggle)](https://www.kaggle.com/datasets/davidpbriggs/kenya-finaccess-household-survey-2024) — curated by iLab Africa
- **Size:** 20,871 respondents across all 47 counties, 28 columns
- **Target:** `financial_status` (Improved / Stayed the same / Worsened)

> The raw dataset is **not** redistributed here — download it from the Kaggle link above and place `finaccess2024_datasprint.csv` alongside the notebook.

## Approach

Following the CRISP-DM framework:

1. **Data cleaning** — handled `barriers_bank` missingness as a meaningful "No barrier" category (these are respondents who already have a bank), standardised inconsistent `education_level` and `marital_status` categories, removed duplicates.
2. **Feature engineering** — income transforms and bands, a **Financial Resilience Index** (0–5 count of resilience indicators), savings-combination and product-use flags.
3. **Modelling** — six models compared: Logistic Regression, Decision Tree, Random Forest, tuned Random Forest (GridSearch), XGBoost, and CatBoost.
4. **Evaluation** — weighted F1, full classification report, confusion matrix, and error analysis.
5. **Interpretation** — feature importances plus Logistic Regression coefficients for directional insight.

## Results

| Model | Weighted F1 |
|---|---|
| **Random Forest (Tuned)** | **0.546** |
| CatBoost | 0.539 |
| Random Forest | 0.531 |
| XGBoost | 0.529 |
| Logistic Regression | 0.526 |
| Decision Tree | 0.493 |

A weighted F1 of ~0.55 reflects genuine noise in self-reported financial status rather than a modelling failure — the model predicts all three classes (it does not collapse to the majority), uses an imbalance-aware metric, and produces stable, interpretable drivers.

## Key Drivers of Financial Deterioration

1. **Monthly income** — the strongest single predictor; lower income raises the risk of worsening.
2. **Financial Resilience Index** — the engineered 0–5 vulnerability score outperforms any single survey question.
3. **Household size** — larger households worsen more often (dependency burden).
4. **Number of financial products held** — thin financial engagement tracks with deterioration.
5. **Financial shocks** — experiencing a shock lifts the worsening rate from 47.2% to 59.6%.

Directional checks confirm: the food-insecure worsen at 62.8% vs 43.0%, and those with debt stress at 63.4% vs 46.3%. Education is protective (university-educated worsen at 38.6% vs 57.3% for no schooling).

## Repository Structure

```
SDC-Datasprint/
├── README.md
├── requirements.txt
├── notebook.ipynb          # full solution: cleaning → EDA → modelling → interpretation
├── models/
│   ├── finaccess_rf_model.pkl    # tuned Random Forest (compressed)
│   ├── scaler.pkl                # fitted StandardScaler
│   └── feature_columns.pkl       # encoded feature order for inference
└── slides/                 # 7-slide presentation
```

## Reproducing

```bash
pip install -r requirements.txt
```

Open `notebook.ipynb` in Jupyter or Google Colab, download the dataset from Kaggle, and run all cells. Pinned library versions matter — the saved model is version-sensitive.

[Link to the Live Server](https://sdc-datasprint-5at.streamlit.app/)

## Author

**Tijani Tatu** — BSc Informatics & Computer Science, Strathmore University
**Charles Maina** — BSc Informatics & Computer Science, Strathmore University
**Tevin Omondi** — BSc Informatics & Computer Science, Strathmore University

