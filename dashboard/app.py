"""
DataSprint 2026 — Predicting Financial Status in Kenya
Streamlit findings dashboard.

Runs entirely on the saved artifacts in ../models/ plus hardcoded summary
numbers from the analysis — the raw FinAccess dataset is NOT required (and is
gitignored). The "Try the Model" tab replicates the notebook's exact
preprocessing (resilience_index, income_band, get_dummies(drop_first=True),
StandardScaler on the 4 numeric columns) before calling model.predict.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Constants & styling
# ----------------------------------------------------------------------------
PRIMARY = "#7A1F2B"   # maroon (slides)
ACCENT = "#B85042"    # warm accent (slides)
NEUTRAL = "#C7B7A3"   # muted contrast for "good" / comparison bars

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# LabelEncoder mapped the target alphabetically: Improved=0, Stayed=1, Worsened=2
CLASS_LABELS = {0: "Improved", 1: "Stayed the same", 2: "Worsened"}
CLASS_COLORS = {"Improved": NEUTRAL, "Stayed the same": ACCENT, "Worsened": PRIMARY}

# The exact numeric columns the saved StandardScaler was fit on (verified from
# scaler.feature_names_in_). Order matters.
NUMERIC_COLS = ["household_size", "monthly_income", "prodsum1", "resilience_index"]

# The five 0/1 indicators summed into the engineered resilience_index (notebook
# Phase 5.2b: (df[resilience_cols] == 'Yes').sum(axis=1)). Each also becomes its
# own *_Yes dummy in the encoded feature matrix.
RESILIENCE_COLS = ["nfhi_11", "nfhi_12", "nfhi_13", "accessto_13k_1month", "not_difficult"]

st.set_page_config(
    page_title="Kenya FinAccess 2026 — Financial Status Dashboard",
    page_icon="📊",
    layout="wide",
)

# Light maroon theming on top of the default theme.
st.markdown(
    f"""
    <style>
      h1, h2, h3 {{ color: {PRIMARY}; }}
      .stTabs [aria-selected="true"] {{ color: {PRIMARY}; }}
      div[data-testid="stMetricValue"] {{ color: {PRIMARY}; }}
      .rec-card {{
        background: #FBF6F4; border-left: 6px solid {PRIMARY};
        border-radius: 8px; padding: 1.1rem 1.3rem; height: 100%;
        color: #2B2B2B;
      }}
      .rec-card h4 {{ color: {PRIMARY}; margin-top: 0; }}
      .rec-card p {{ color: #2B2B2B; }}
      .disclaimer {{
        background: #FFF4E5; border-left: 6px solid {ACCENT};
        border-radius: 8px; padding: 0.9rem 1.1rem;
        color: #2B2B2B;
      }}
      .disclaimer b {{ color: {PRIMARY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "finaccess_rf_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
    return model, scaler, list(feature_columns)


def hbar(labels, values, colors, title, xtitle, value_fmt="{:.3f}"):
    """A simple horizontal bar chart (largest at top)."""
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[value_fmt.format(v) for v in values],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=360,
        plot_bgcolor="white",
    )
    return fig


def grouped_worsened_chart(rows):
    """rows: list of (group_name, fragile_label, fragile_val, safe_label, safe_val)."""
    fig = go.Figure()
    fig.add_bar(
        name="Fragile group",
        y=[r[0] for r in rows],
        x=[r[2] for r in rows],
        orientation="h",
        marker_color=PRIMARY,
        text=[f"{r[2]:.1f}%" for r in rows],
        textposition="auto",
        customdata=[r[1] for r in rows],
        hovertemplate="%{customdata}: %{x:.1f}%<extra></extra>",
    )
    fig.add_bar(
        name="Comparison group",
        y=[r[0] for r in rows],
        x=[r[4] for r in rows],
        orientation="h",
        marker_color=NEUTRAL,
        text=[f"{r[4]:.1f}%" for r in rows],
        textposition="auto",
        customdata=[r[3] for r in rows],
        hovertemplate="%{customdata}: %{x:.1f}%<extra></extra>",
    )
    fig.update_layout(
        barmode="group",
        title="% Worsened by group (fragility crosstabs)",
        xaxis_title="% reporting financial situation Worsened",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ----------------------------------------------------------------------------
# Prediction pipeline (mirrors the notebook exactly)
# ----------------------------------------------------------------------------
def income_to_band(monthly_income: float) -> str:
    """Replicate income_band = pd.qcut(monthly_income, q=3, labels=['Low','Mid','High']).

    NOTE: the exact tertile cut points were computed on the full training set,
    which is gitignored and not shipped with the dashboard, so they cannot be
    recovered exactly here. These thresholds are approximate stand-ins for an
    *illustrative* prediction. The model also sees income directly via the
    scaled `monthly_income` numeric, which carries the main income signal.
    """
    if monthly_income <= 5000:
        return "Low"
    if monthly_income <= 15000:
        return "Mid"
    return "High"


def build_input_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    """Build a single-row, fully-aligned encoded feature frame.

    Strategy: start from all-zeros over the exact encoded columns (every dummy
    at 0 == the drop_first reference category), then switch on the columns the
    user has specified. Anything not exposed as a widget stays at its baseline.
    """
    row = {c: 0 for c in feature_columns}

    # --- numeric features ---
    row["household_size"] = inputs["household_size"]
    row["monthly_income"] = inputs["monthly_income"]
    row["prodsum1"] = inputs["prodsum1"]

    # --- resilience: index (numeric) + the 5 underlying *_Yes dummies ---
    resilience_index = 0
    for col in RESILIENCE_COLS:
        if inputs["resilience"].get(col):
            resilience_index += 1
            dummy = f"{col}_Yes"
            if dummy in row:
                row[dummy] = 1
    row["resilience_index"] = resilience_index

    # --- financial shock ---
    if inputs["experienced_shock"]:
        row["experienced_shock_Yes"] = 1

    # --- income band (Low is the dropped reference -> Mid / High dummies) ---
    band = income_to_band(inputs["monthly_income"])
    if band == "Mid" and "income_band_Mid" in row:
        row["income_band_Mid"] = 1
    elif band == "High" and "income_band_High" in row:
        row["income_band_High"] = 1

    # --- optional demographics (each maps to one dummy column if selected) ---
    for col in (inputs.get("sex"), inputs.get("location_type"),
                inputs.get("age"), inputs.get("education")):
        if col and col in row:
            row[col] = 1

    # Align to the exact column order; verify before returning.
    df = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    return df, band, resilience_index


def predict(model, scaler, feature_columns, df: pd.DataFrame):
    # Pipeline integrity check (task requirement): exact columns, exact order.
    assert list(df.columns) == list(feature_columns), "Encoded columns do not match feature_columns."
    assert df.shape[1] == model.n_features_in_, "Column count != model.n_features_in_."

    scaled = df.astype(float).copy()
    scaled[NUMERIC_COLS] = scaler.transform(scaled[NUMERIC_COLS])

    pred = int(model.predict(scaled)[0])
    proba = model.predict_proba(scaled)[0]
    return pred, proba


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("Predicting Financial Status in Kenya")
st.caption(
    "2024 FinAccess Household Survey · 20,871 adults · Strathmore Data Community × iLab Africa DataSprint 2026"
)

tab_overview, tab_drivers, tab_reco, tab_model = st.tabs(
    ["📌 Overview", "📉 What Drives Deterioration", "✅ Recommendations", "🧪 Try the Model"]
)

# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================
with tab_overview:
    st.header("The problem")
    st.markdown(
        "Despite a decade of progress in financial inclusion, the 2024 FinAccess "
        "Household Survey found that **9.9% of Kenyan adults remain fully excluded** "
        "from financial services, and **over half report their financial situation "
        "has worsened** year-on-year. This project builds a multiclass classifier to "
        "predict financial status and — more importantly — to identify *which factors "
        "drive financial deterioration* so policymakers, banks, and NGOs know what to "
        "prioritise."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Adults surveyed", "20,871")
    c2.metric("Reported worsened", "52.6%")
    c3.metric("Fully excluded", "9.9%")
    c4.metric("Best model (weighted F1)", "0.546")

    st.divider()
    st.subheader("Target distribution — `financial_status`")
    left, right = st.columns([3, 2])
    with left:
        labels = ["Worsened", "Stayed the same", "Improved"]
        values = [52.6, 26.9, 20.5]
        fig = hbar(
            labels,
            values,
            [CLASS_COLORS[l] for l in labels],
            "Share of respondents by financial status (%)",
            "% of respondents",
            value_fmt="{:.1f}%",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(
            "This is an **imbalanced multiclass** problem, so we evaluate on "
            "**weighted F1**, not accuracy — a model that always predicted "
            "*Worsened* would be ~53% accurate but useless for the other two "
            "classes.\n\n"
            "- **Worsened** — 52.6%\n"
            "- **Stayed the same** — 26.9%\n"
            "- **Improved** — 20.5%"
        )

# ============================================================================
# TAB 2 — WHAT DRIVES DETERIORATION
# ============================================================================
with tab_drivers:
    st.header("What drives deterioration")
    st.markdown(
        "Two complementary views: what the **tuned Random Forest** relies on "
        "(feature importances), and how the worsening rate splits across "
        "**fragility groups** in the raw crosstabs."
    )

    col_left, col_right = st.columns(2)

    with col_left:
        feat_labels = [
            "Monthly income",
            "Resilience Index",
            "Household size",
            "No. of products",
            "Financial shock",
        ]
        feat_vals = [0.076, 0.054, 0.052, 0.049, 0.019]
        # Darkest for the top driver, fading down.
        feat_colors = [PRIMARY, "#8E2E39", ACCENT, "#C56A5C", "#D49488"]
        st.plotly_chart(
            hbar(feat_labels, feat_vals, feat_colors,
                 "Top feature importances (tuned Random Forest)",
                 "Importance", value_fmt="{:.3f}"),
            use_container_width=True,
        )

    with col_right:
        rows = [
            ("Food (in)security", "Food insecure", 62.8, "Food secure", 43.0),
            ("Debt stress", "Has debt stress", 63.4, "No debt stress", 46.3),
            ("Financial shock", "Shock: Yes", 59.6, "Shock: No", 47.2),
        ]
        st.plotly_chart(grouped_worsened_chart(rows), use_container_width=True)

    st.info(
        "**Read-out.** Income is the single strongest predictor, but the engineered "
        "**Financial Resilience Index** (food security + debt-stress freedom + "
        "emergency-fund access) ranks second — vulnerability is *cumulative*, not "
        "down to any one factor. The crosstabs confirm direction: the food-insecure, "
        "debt-stressed, and shock-affected all worsen ~13–20 points more often than "
        "their counterparts."
    )

# ============================================================================
# TAB 3 — RECOMMENDATIONS
# ============================================================================
with tab_reco:
    st.header("Recommendations")
    st.caption("Pulled from the notebook's Phase 10 — Conclusions & Recommendations.")

    cards = [
        (
            "Policymakers / NGOs",
            "Target the <b>structurally vulnerable</b>, not just shock victims. "
            "Larger, low-income households and lagging counties (Kisumu, West Pokot "
            "ranked among predictors) are most exposed. Shock-responsive safety nets "
            "matter — a shock adds ~12 points to the worsening rate — but the "
            "resilience index shows vulnerability builds <b>cumulatively</b>, so "
            "prioritise emergency-fund and food-security support <b>before</b> a "
            "shock hits rather than only reacting after.",
        ),
        (
            "Banks / SACCOs / MFIs",
            "Income and thin product use predict deterioration, so deepening "
            "financial engagement is protective. The bank-barrier data shows "
            "<b>affordability and service quality</b> drive worsening (58–72%), while "
            "eligibility does not — cost and trust, not qualification rules, keep "
            "at-risk Kenyans underserved. Pair affordable, low-fee products with "
            "credit restructuring rather than withdrawing access.",
        ),
        (
            "NGOs / mobile-money providers",
            "Education and financial literacy track strongly with better outcomes "
            "(university-educated worsen at 38.6% vs 57.3% for no schooling). Attach "
            "<b>financial-literacy programmes to mobile-money platforms</b> to reach "
            "low-education households, and extend low-cost emergency liquidity to "
            "those who cannot raise KES 13,000 in a month — the group the resilience "
            "index flags as most fragile.",
        ),
    ]
    cols = st.columns(3)
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="rec-card"><h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

# ============================================================================
# TAB 4 — TRY THE MODEL (ILLUSTRATIVE)
# ============================================================================
with tab_model:
    st.header("Try the model")

    st.markdown(
        '<div class="disclaimer"><b>⚠️ Illustrative only — not financial advice.</b><br>'
        "The tuned Random Forest scores a weighted F1 of ~0.55, which reflects genuine "
        "noise in self-reported financial status. Predictions below are a teaching aid "
        "to show how the model reasons about the key drivers — they are <b>not</b> a "
        "reliable assessment of any real person's finances, and unspecified survey "
        "fields are held at their baseline (reference) values.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    try:
        model, scaler, feature_columns = load_artifacts()
    except Exception as e:  # pragma: no cover - surfaced in UI
        st.error(f"Could not load model artifacts from {MODELS_DIR}: {e}")
        st.stop()

    with st.form("predict_form"):
        st.subheader("Key financial drivers")
        a, b = st.columns(2)
        with a:
            monthly_income = st.slider(
                "Monthly income (KES)", min_value=0, max_value=100000,
                value=12000, step=500,
                help="Strongest single predictor. Also drives the Low/Mid/High income band.",
            )
            household_size = st.slider("Household size", 1, 20, 4)
        with b:
            prodsum1 = st.slider(
                "Number of financial products held", 0, 12, 2,
                help="prodsum1 — more products tracks with greater resilience.",
            )
            experienced_shock = st.checkbox(
                "Experienced a financial shock in the past year",
                help="Lifts the worsening rate from 47.2% to 59.6% in the data.",
            )

        st.subheader("Financial resilience indicators")
        st.caption(
            "Each ticked box adds 1 to the 0–5 Financial Resilience Index "
            "(and sets the matching survey flag)."
        )
        r1, r2, r3 = st.columns(3)
        resilience = {}
        resilience["accessto_13k_1month"] = r1.checkbox(
            "Could raise KES 13,000 for an emergency within a month")
        resilience["not_difficult"] = r2.checkbox(
            "Meeting monthly expenses is not difficult")
        resilience["nfhi_11"] = r3.checkbox(
            "Food secure (did not run short of food)", help="Survey item nfhi_11")
        resilience["nfhi_12"] = r1.checkbox(
            "Free of debt-repayment stress", help="Survey item nfhi_12")
        resilience["nfhi_13"] = r2.checkbox(
            "Able to keep up with bills/commitments", help="Survey item nfhi_13")

        st.subheader("Demographics (optional)")
        d1, d2, d3, d4 = st.columns(4)
        sex_choice = d1.radio("Sex", ["Female (baseline)", "Male"], index=0)
        loc_choice = d2.radio("Location", ["Rural (baseline)", "Urban"], index=0)
        age_choice = d3.selectbox(
            "Age band",
            ["Baseline", "18-25", "26-35", "36-45", "46-55", "Above 55"],
            index=2,
        )
        edu_choice = d4.selectbox(
            "Education level",
            ["Baseline", "None", "Some primary", "Primary completed",
             "Some secondary", "Secondary completed", "Some university",
             "University completed"],
            index=0,
        )

        submitted = st.form_submit_button("Predict financial status", type="primary")

    if submitted:
        inputs = {
            "monthly_income": float(monthly_income),
            "household_size": float(household_size),
            "prodsum1": float(prodsum1),
            "experienced_shock": experienced_shock,
            "resilience": resilience,
            "sex": "Sex_Male" if sex_choice == "Male" else None,
            "location_type": "location_type_Urban" if loc_choice == "Urban" else None,
            "age": f"Age_{age_choice}" if age_choice != "Baseline" else None,
            "education": f"education_level_{edu_choice}" if edu_choice != "Baseline" else None,
        }

        df, band, resilience_index = build_input_row(inputs, feature_columns)

        try:
            pred, proba = predict(model, scaler, feature_columns, df)
        except AssertionError as e:
            st.error(f"Pipeline integrity check failed: {e}")
            st.stop()

        label = CLASS_LABELS[pred]
        st.divider()
        st.subheader("Illustrative prediction")
        res_col, prob_col = st.columns([2, 3])
        with res_col:
            st.metric("Most likely financial status", label)
            st.caption(
                f"Income band: **{band}** · Resilience Index: **{resilience_index}/5** · "
                f"Verified {df.shape[1]} encoded features against the saved schema."
            )
        with prob_col:
            order = ["Worsened", "Stayed the same", "Improved"]
            idx = {v: k for k, v in CLASS_LABELS.items()}
            vals = [round(proba[idx[o]] * 100, 1) for o in order]
            st.plotly_chart(
                hbar(order, vals, [CLASS_COLORS[o] for o in order],
                     "Predicted class probabilities", "Probability (%)",
                     value_fmt="{:.1f}%"),
                use_container_width=True,
            )
        st.caption(
            "Reminder: illustrative output from a modest-F1 model — not financial advice."
        )
