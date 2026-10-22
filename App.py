import html
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1020;
    --panel: #121a30;
    --panel-2: #18223c;
    --border: rgba(148, 163, 184, 0.16);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --brand: #818cf8;
    --good: #34d399;
    --warn: #fbbf24;
    --bad: #f87171;
}

.stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
.stApp label, .stApp input, .stApp textarea, .stApp button {
    font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
}

.block-container { padding-top: 2.4rem; max-width: 1400px; }
footer { visibility: hidden; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #0e1428;
    border-right: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 18px; }
.brand-logo {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    display: grid; place-items: center; font-size: 1.4rem;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4);
}
.brand-name { font-weight: 800; font-size: 1.1rem; color: #fff; line-height: 1.1; }
.brand-sub { font-size: 0.75rem; color: var(--muted); }
.side-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--muted); margin: 18px 0 6px;
}
.pill {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 10px; margin-bottom: 6px;
    font-size: 0.85rem; font-weight: 500;
}
.pill .dot { width: 8px; height: 8px; border-radius: 50%; }
.pill-ok { background: rgba(52, 211, 153, 0.1); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.25); }
.pill-ok .dot { background: var(--good); box-shadow: 0 0 8px var(--good); }
.pill-err { background: rgba(248, 113, 113, 0.1); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.25); }
.pill-err .dot { background: var(--bad); }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden;
    background:
        radial-gradient(900px 260px at 0% 0%, rgba(129, 140, 248, 0.30), transparent 60%),
        radial-gradient(600px 240px at 100% 100%, rgba(168, 85, 247, 0.20), transparent 60%),
        linear-gradient(135deg, #111936 0%, #0d1329 100%);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 28px 32px; margin-bottom: 24px;
}
.hero-title { font-size: 2rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; margin-bottom: 6px; }
.hero-sub { color: #aab4c8; font-size: 1rem; margin-bottom: 16px; max-width: 760px; }
.chip {
    display: inline-block; padding: 4px 12px; margin: 0 8px 6px 0;
    border-radius: 999px; font-size: 0.76rem; font-weight: 600;
    background: rgba(129, 140, 248, 0.14); color: #c7d2fe;
    border: 1px solid rgba(129, 140, 248, 0.3);
}

/* ---------- Section titles ---------- */
.sec { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.02rem; color: #fff; margin: 6px 0 12px; }
.sec .n {
    width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6); font-size: 0.8rem; color: #fff;
}
.page-title { font-size: 1.5rem; font-weight: 800; color: #fff; margin: 0 0 4px; }
.page-sub { color: var(--muted); margin-bottom: 18px; }

/* ---------- KPI tiles ---------- */
.kpi {
    background: var(--panel); border: 1px solid var(--border);
    border-top: 3px solid var(--accent, var(--brand));
    border-radius: 16px; padding: 18px 20px; height: 100%;
}
.kpi .lbl { font-size: 0.74rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
.kpi .val { font-size: 1.9rem; font-weight: 800; color: var(--accent, #fff); margin-top: 4px; line-height: 1.15; }
.kpi .sub { font-size: 0.82rem; color: var(--muted); margin-top: 2px; }

/* ---------- Stacked risk bar ---------- */
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px; }
.stack { display: flex; height: 14px; border-radius: 999px; overflow: hidden; background: rgba(255, 255, 255, 0.06); margin: 10px 0 12px; }
.legend { display: flex; gap: 18px; flex-wrap: wrap; font-size: 0.85rem; color: var(--muted); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }

/* ---------- Result panel ---------- */
.result {
    background: linear-gradient(160deg, var(--panel-2), var(--panel));
    border: 1px solid var(--accent); border-radius: 20px; padding: 24px;
    box-shadow: 0 10px 40px -12px var(--accent);
}
.result-top { display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }
.gauge {
    width: 150px; height: 150px; border-radius: 50%; flex: none;
    background: conic-gradient(var(--c) calc(var(--p) * 1%), rgba(255, 255, 255, 0.08) 0);
    display: grid; place-items: center;
}
.gauge-inner {
    width: 116px; height: 116px; border-radius: 50%; background: #121a30;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.gauge-val { font-size: 2rem; font-weight: 800; color: #fff; line-height: 1; }
.gauge-lbl { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; }
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
    color: #0b1020; background: var(--accent);
}
.verdict-title { font-size: 1.35rem; font-weight: 800; color: #fff; margin: 10px 0 4px; line-height: 1.2; }
.verdict-sub { color: var(--muted); font-size: 0.92rem; }
.verdict-sub code { background: rgba(255, 255, 255, 0.08); color: #e2e8f0; padding: 1px 6px; border-radius: 6px; }
.mini { display: flex; align-items: center; gap: 10px; margin-top: 14px; font-size: 0.85rem; color: var(--muted); }
.mini span { width: 78px; }
.mini b { width: 52px; text-align: right; color: #fff; }
.track { flex: 1; height: 8px; border-radius: 999px; background: rgba(255, 255, 255, 0.08); overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.action {
    margin-top: 18px; padding: 12px 14px; border-radius: 12px; font-size: 0.9rem; color: #dbe3f1;
    background: rgba(255, 255, 255, 0.04); border: 1px solid var(--border);
}
.snap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 16px; }
.snap div { background: rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 8px 10px; }
.snap small { display: block; color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; }
.snap b { color: #fff; font-size: 0.92rem; }

.insight {
    display: flex; gap: 12px; align-items: flex-start; padding: 12px 14px; margin-bottom: 8px;
    background: var(--panel); border: 1px solid var(--border); border-left: 4px solid var(--lvl);
    border-radius: 12px; font-size: 0.9rem; color: #dbe3f1;
}
.insight b { color: #fff; }
.insight .m { color: var(--muted); font-size: 0.84rem; }

.placeholder {
    border: 2px dashed rgba(148, 163, 184, 0.28); border-radius: 20px; padding: 46px 24px; text-align: center;
    background: rgba(255, 255, 255, 0.02);
}
.ph-ico { font-size: 2.6rem; margin-bottom: 8px; }
.ph-title { font-weight: 700; color: #fff; font-size: 1.1rem; margin-bottom: 6px; }
.ph-sub { color: var(--muted); font-size: 0.9rem; }

/* ---------- Model & insights ---------- */
.fi-row { display: flex; align-items: center; gap: 12px; margin: 9px 0; font-size: 0.88rem; color: #dbe3f1; }
.fi-row .nm { width: 210px; flex: none; }
.fi-row .tr { flex: 1; height: 10px; border-radius: 999px; background: rgba(255, 255, 255, 0.06); overflow: hidden; }
.fi-row .fl { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #a855f7); }
.fi-row .pc { width: 56px; text-align: right; color: var(--muted); }

/* ---------- Buttons ---------- */
[data-testid^="stBaseButton-primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6); border: 0; color: #fff; font-weight: 700;
    padding: 0.65rem 1rem; border-radius: 12px; box-shadow: 0 8px 22px -8px rgba(99, 102, 241, 0.7);
}
[data-testid^="stBaseButton-primary"]:hover { filter: brightness(1.1); }
</style>
""",
    unsafe_allow_html=True,
)


def md(markup):
    """Render HTML safely: collapse indentation so Markdown never treats it as a code block."""
    st.markdown(" ".join(line.strip() for line in markup.strip().splitlines()), unsafe_allow_html=True)


def section(number, title):
    md(f'<div class="sec"><span class="n">{number}</span>{title}</div>')


# ---------------------------------------------------------
# Model Loading & Caching
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

MODEL_PATHS = {
    "Random Forest": APP_DIR / "rf_model.pkl",
    "XGBoost": APP_DIR / "xgb_model.json",
}
XGB_PREPROCESSOR_PATH = APP_DIR / "xgb_preprocessor.pkl"
SAMPLE_CSV_PATH = APP_DIR / "sample_customers_test.csv"
CLEAN_DATA_PATH = APP_DIR / "customer_churn_clean_data.csv"

# Required features in expected order
EXPECTED_NUM_COLS = [
    'Age', 'Tenure', 'Usage Frequency', 'Support Calls',
    'Payment Delay', 'Total Spend', 'Last Interaction'
]
EXPECTED_CAT_COLS = ['Gender', 'Subscription Type', 'Contract Length']

FEATURE_ALIASES = {
    'Age': ['age', 'Age'],
    'Gender': ['gender', 'Gender', 'sex', 'Sex'],
    'Tenure': ['tenure', 'Tenure', 'tenure_months', 'Tenure Months'],
    'Usage Frequency': ['usage frequency', 'Usage Frequency', 'usage_frequency', 'UsageFrequency', 'usage'],
    'Support Calls': ['support calls', 'Support Calls', 'support_calls', 'SupportCalls', 'support'],
    'Payment Delay': ['payment delay', 'Payment Delay', 'payment_delay', 'PaymentDelay', 'delay'],
    'Subscription Type': ['subscription type', 'Subscription Type', 'subscription_type', 'SubscriptionType', 'plan'],
    'Contract Length': ['contract length', 'Contract Length', 'contract_length', 'ContractLength', 'contract'],
    'Total Spend': ['total spend', 'Total Spend', 'total_spend', 'TotalSpend', 'spend'],
    'Last Interaction': ['last interaction', 'Last Interaction', 'last_interaction', 'LastInteraction', 'interaction']
}

RISK_COLORS = {"High Risk": "#f87171", "Medium Risk": "#fbbf24", "Low Risk": "#34d399"}


@st.cache_resource(show_spinner="Loading prediction models...")
def load_churn_model(model_name):
    """Load a saved model and return an explicit error when it is unavailable."""
    model_path = MODEL_PATHS[model_name]
    try:
        if model_name == "XGBoost":
            if not model_path.is_file():
                return None, None, f"Model file `{model_path.name}` was not found."
            if not XGB_PREPROCESSOR_PATH.is_file():
                return (
                    None,
                    None,
                    f"Preprocessor file `{XGB_PREPROCESSOR_PATH.name}` was not found.",
                )
            model = XGBClassifier()
            model.load_model(model_path)
            preprocessor = joblib.load(XGB_PREPROCESSOR_PATH)
            return model, preprocessor, None

        if not model_path.is_file():
            return None, None, f"Model file `{model_path.name}` was not found."
        model = joblib.load(model_path)
        if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            model.named_steps['classifier'].n_jobs = 1
        return model, None, None
    except Exception as e:
        return None, None, f"Could not load `{model_path.name}`: {type(e).__name__}: {e}"


loaded_models = {
    model_name: load_churn_model(model_name)
    for model_name in MODEL_PATHS
}


@st.cache_data(show_spinner=False)
def load_dataset_summary():
    """Return (customer count, churn rate %) from the training data, or None."""
    try:
        churn = pd.read_csv(CLEAN_DATA_PATH, usecols=['Churn'])['Churn']
        return int(len(churn)), float(churn.mean() * 100)
    except Exception:
        return None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    md("""
    <div class="brand">
        <div class="brand-logo">🔮</div>
        <div><div class="brand-name">Churn Predictor</div><div class="brand-sub">NTI ML Project</div></div>
    </div>
    """)

    md('<div class="side-label">Navigation</div>')
    app_mode = st.radio(
        "Choose Mode:",
        ["📁 Batch CSV Prediction", "👤 Single Customer Prediction", "📊 Model & Insights"],
        index=0,
        label_visibility="collapsed",
    )

    md('<div class="side-label">Prediction model</div>')
    selected_model_name = st.selectbox(
        "Choose a model:",
        list(MODEL_PATHS),
        label_visibility="collapsed",
    )
    model, model_preprocessor, model_error = loaded_models[selected_model_name]

    md('<div class="side-label">System status</div>')
    for model_name, (loaded_model, _, load_error) in loaded_models.items():
        if loaded_model is not None:
            md(f'<div class="pill pill-ok"><span class="dot"></span>{model_name} loaded</div>')
        else:
            md(f'<div class="pill pill-err"><span class="dot"></span>{model_name} unavailable</div>')
            st.caption(load_error)

    md('<div class="side-label">Quick download</div>')

    # Provide sample CSV template
    sample_template = pd.DataFrame({
        'CustomerID': [101, 102],
        'Name': ['Ali Ahmed', 'Fatma Omar'],
        'Age': [35, 60],
        'Gender': ['Male', 'Female'],
        'Tenure': [24, 6],
        'Usage Frequency': [18, 4],
        'Support Calls': [1, 8],
        'Payment Delay': [3, 20],
        'Subscription Type': ['Standard', 'Basic'],
        'Contract Length': ['Annual', 'Monthly'],
        'Total Spend': [750.0, 220.0],
        'Last Interaction': [10, 28]
    })
    csv_template_bytes = sample_template.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download CSV Template",
        data=csv_template_bytes,
        file_name="churn_prediction_template.csv",
        mime="text/csv",
        width="stretch"
    )

    st.markdown("---")
    st.caption("NTI Machine Learning Project\n\nDeveloped with Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
md(f"""
<div class="hero">
    <div class="hero-title">🔮 Customer Churn Prediction</div>
    <div class="hero-sub">Spot customers who are likely to leave, understand why, and act before they do.</div>
    <span class="chip">Model: {html.escape(selected_model_name)}</span>
    <span class="chip">Single &amp; batch predictions</span>
    <span class="chip">Risk levels &amp; insights</span>
</div>
""")


# ---------------------------------------------------------
# Helper Functions for Data Processing
# ---------------------------------------------------------
def find_column_by_aliases(df, aliases):
    """Find matching column name in dataframe case-insensitively."""
    cols_lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in cols_lower:
            return cols_lower[alias.lower()]
    return None


def get_churn_probabilities(trained_model, features):
    """Return probabilities for class 1, or None if the model has no probability API."""
    predict_proba = getattr(trained_model, "predict_proba", None)
    if not callable(predict_proba):
        return None

    classes = list(getattr(trained_model, "classes_", []))
    churn_class_index = next(
        (index for index, class_value in enumerate(classes) if class_value == 1),
        None,
    )
    if churn_class_index is None:
        raise ValueError("The loaded model does not contain churn class 1.")

    return predict_proba(features)[:, churn_class_index]


def prepare_model_features(features, preprocessor):
    """Apply a separately saved preprocessor when the classifier needs one."""
    return preprocessor.transform(features) if preprocessor is not None else features


def risk_level(prob_percent):
    """Map a churn probability (0-100) to a risk band."""
    if prob_percent >= 65:
        return "High Risk"
    if prob_percent >= 35:
        return "Medium Risk"
    return "Low Risk"


def preprocess_and_predict(df_input, trained_model, preprocessor=None):
    """Prepares dataframe, finds ID/Name, formats features, and runs prediction."""
    df = df_input.copy()
    if df.empty:
        return None, None, "The input CSV contains no customer rows."

    generated_ids = pd.Series(
        [f"CUST-{i+1:04d}" for i in range(len(df))], index=df.index
    )

    # Identify Customer ID column
    id_col = find_column_by_aliases(df, ['CustomerID', 'Customer_ID', 'ID', 'Id', 'id', 'CID'])
    if id_col is not None:
        id_series = df[id_col].fillna("").astype(str).str.strip()
        id_series = id_series.mask(id_series.eq(""), generated_ids)
    else:
        id_series = generated_ids

    # Identify Name column
    name_col = find_column_by_aliases(df, ['Name', 'Customer Name', 'Customer_Name', 'FullName', 'Full Name', 'name'])
    if name_col is not None:
        name_series = (
            df[name_col]
            .fillna("")
            .astype(str)
            .str.strip()
            .replace("", "Not provided")
        )
    else:
        name_series = pd.Series([f"Customer {cid}" for cid in id_series], index=df.index)

    # Match required features
    matched_features = {}
    missing_cols = []
    for feat, aliases in FEATURE_ALIASES.items():
        matched = find_column_by_aliases(df, aliases)
        if matched is not None:
            matched_features[feat] = df[matched]
        else:
            missing_cols.append(feat)

    if missing_cols:
        return None, None, f"Missing required columns: {', '.join(missing_cols)}"

    # Construct exact features dataframe
    X = pd.DataFrame(matched_features)

    # Validate feature values before passing them to the model.
    for col in EXPECTED_NUM_COLS:
        values = pd.to_numeric(X[col], errors='coerce')
        invalid_values = values.isna() | ~np.isfinite(values.to_numpy())
        if invalid_values.any():
            row_numbers = ", ".join(
                str(index + 2)
                for index in np.flatnonzero(invalid_values.to_numpy())[:5]
            )
            return (
                None,
                None,
                f"'{col}' must contain a finite number in every row. "
                f"Check CSV row(s): {row_numbers}.",
            )
        X[col] = values

    for col in EXPECTED_CAT_COLS:
        values = X[col].fillna("").astype(str).str.strip()
        missing_values = values.eq("")
        if missing_values.any():
            row_numbers = ", ".join(
                str(index + 2)
                for index in np.flatnonzero(missing_values.to_numpy())[:5]
            )
            return (
                None,
                None,
                f"'{col}' is required in every row. Check CSV row(s): {row_numbers}.",
            )
        X[col] = values

    # Make predictions
    model_features = prepare_model_features(X, preprocessor)
    preds = trained_model.predict(model_features)
    probs = get_churn_probabilities(trained_model, model_features)

    # Build clean output dataframe with ID and Name
    results_df = pd.DataFrame({
        'CustomerID': id_series,
        'Name': name_series,
        'Prediction': np.where(preds == 1, 'Churn (At Risk)', 'Retained (Active)'),
    })
    if probs is not None:
        results_df['Churn Probability (%)'] = (probs * 100).round(2)
        results_df['Risk Level'] = np.where(
            probs >= 0.65,
            'High Risk',
            np.where(probs >= 0.35, 'Medium Risk', 'Low Risk'),
        )

    # Also attach full features for complete export
    full_export_df = pd.concat([results_df, X], axis=1)
    return results_df, full_export_df, None


def build_insights(age, tenure, usage_freq, support_calls, payment_delay, contract_length):
    """Return a list of (level, title, detail) insight tuples for a single customer."""
    insights = []
    if payment_delay >= 15:
        insights.append(("warn", "High payment delay", f"{payment_delay} days late. Long delays strongly correlate with churn."))
    if support_calls >= 5:
        insights.append(("warn", "Frequent support calls", f"{support_calls} calls. High call volume signals frustration."))
    if contract_length == "Monthly":
        insights.append(("warn", "Monthly contract", "Month-to-month customers churn more than annual ones."))
    if usage_freq <= 5:
        insights.append(("warn", "Low usage frequency", f"{usage_freq} times/month. Inactive customers are more likely to leave."))
    if tenure <= 6:
        insights.append(("info", "Short tenure", f"{tenure} months. New customers need onboarding care."))
    if not insights:
        insights.append(("good", "Healthy account", "Low payment delay, reasonable support volume and steady activity."))
    return insights


def render_result(res):
    """Render the prediction result panel for a single customer."""
    name = html.escape(res["name"])
    cid = html.escape(res["id"])
    prob = res["prob"]
    is_churn = res["pred"] == 1

    if prob is not None:
        level = risk_level(prob)
        color = RISK_COLORS[level]
        gauge = (
            f'<div class="gauge" style="--p:{prob:.1f};--c:{color}"><div class="gauge-inner">'
            f'<div class="gauge-val">{prob:.0f}%</div><div class="gauge-lbl">churn risk</div></div></div>'
        )
        bars = (
            f'<div class="mini"><span>Churn</span><div class="track"><div class="fill" style="width:{prob:.1f}%;background:#f87171"></div></div><b>{prob:.1f}%</b></div>'
            f'<div class="mini"><span>Retention</span><div class="track"><div class="fill" style="width:{100 - prob:.1f}%;background:#34d399"></div></div><b>{100 - prob:.1f}%</b></div>'
        )
    else:
        level = "High Risk" if is_churn else "Low Risk"
        color = RISK_COLORS[level]
        gauge = f'<div class="gauge" style="--p:{100 if is_churn else 0};--c:{color}"><div class="gauge-inner"><div class="gauge-val">{"⚠" if is_churn else "✓"}</div></div></div>'
        bars = ""

    title = "Likely to churn" if is_churn else "Likely to stay"
    actions = {
        "High Risk": "Reach out now: assign an account manager and offer a retention incentive or contract upgrade.",
        "Medium Risk": "Keep an eye on this account: schedule a proactive check-in and resolve any open support issues.",
        "Low Risk": "Healthy customer: keep engagement steady and consider an upsell or loyalty reward.",
    }[level]

    inp = res["inputs"]
    snap = "".join(
        f"<div><small>{label}</small><b>{html.escape(str(value))}</b></div>"
        for label, value in inp.items()
    )

    md(f"""
    <div class="result" style="--accent:{color}">
        <div class="result-top">
            {gauge}
            <div style="flex:1;min-width:180px">
                <span class="badge">{level}</span>
                <div class="verdict-title">{title}</div>
                <div class="verdict-sub"><b style="color:#fff">{name}</b> · <code>{cid}</code></div>
            </div>
        </div>
        {bars}
        <div class="action">💡 <b>Suggested action:</b> {actions}</div>
        <div class="snap">{snap}</div>
    </div>
    """)
    st.caption(f"Predicted with {res['model']}")

    md('<div style="height:18px"></div>')
    section("💡", "Risk factors &amp; insights")
    level_colors = {"warn": "#fbbf24", "info": "#818cf8", "good": "#34d399"}
    level_icons = {"warn": "⚠️", "info": "ℹ️", "good": "✨"}
    for lvl, ttl, detail in res["insights"]:
        md(f"""
        <div class="insight" style="--lvl:{level_colors[lvl]}">
            <span>{level_icons[lvl]}</span>
            <div><b>{ttl}</b><div class="m">{detail}</div></div>
        </div>
        """)


def get_feature_importance(trained_model, preprocessor):
    """Return a list of (readable feature name, importance %) sorted high to low, or None."""
    try:
        steps = getattr(trained_model, "named_steps", {})
        classifier = steps.get("classifier", trained_model)
        prep = steps.get("preprocessor", preprocessor)
        importances = getattr(classifier, "feature_importances_", None)
        if importances is None or prep is None:
            return None
        names = []
        for raw in prep.get_feature_names_out():
            kind, _, rest = str(raw).partition("__")
            if kind == "cat" and "_" in rest:
                feature, _, value = rest.rpartition("_")
                names.append(f"{feature}: {value}")
            else:
                names.append(rest or str(raw))
        total = float(np.sum(importances)) or 1.0
        pairs = [(n, float(v) / total * 100) for n, v in zip(names, importances)]
        return sorted(pairs, key=lambda p: p[1], reverse=True)
    except Exception:
        return None


# ---------------------------------------------------------
# MODE 1: Batch CSV Prediction
# ---------------------------------------------------------
if app_mode == "📁 Batch CSV Prediction":
    md("""
    <div class="page-title">📁 Batch customer prediction</div>
    <div class="page-sub">Upload a CSV and get a churn verdict, probability and risk level for every customer.</div>
    """)

    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    with col2:
        use_demo = st.button(
            "🧪 Try with demo data",
            help="Load sample customers to see how it works",
            width="stretch",
            type="primary",
        )

    df_to_predict = None

    if use_demo:
        if SAMPLE_CSV_PATH.is_file():
            df_to_predict = pd.read_csv(SAMPLE_CSV_PATH)
            st.info("Loaded demo test dataset with sample customers.")
        elif CLEAN_DATA_PATH.is_file():
            df_sample = pd.read_csv(CLEAN_DATA_PATH, nrows=50)
            df_sample['Name'] = [f"Customer {int(cid)}" for cid in df_sample['CustomerID']]
            df_to_predict = df_sample
            st.info("Loaded 50 sample customers from dataset.")
        else:
            st.error("Demo data is unavailable. Check that the customer dataset is present.")
    elif uploaded_file is not None:
        try:
            df_to_predict = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded CSV file: {e}")

    if df_to_predict is None:
        md("""
        <div class="placeholder">
            <div class="ph-ico">📄</div>
            <div class="ph-title">No file yet</div>
            <div class="ph-sub">Upload a CSV above, or try the demo data. Need the format? Download the template from the sidebar.</div>
        </div>
        """)
    else:
        with st.expander("👀 Preview uploaded data (first 5 rows)", expanded=False):
            st.dataframe(df_to_predict.head(), width="stretch")

        if model is None:
            st.error(model_error)
        else:
            with st.spinner("Analyzing customer data and predicting churn..."):
                results_df, full_df, error = preprocess_and_predict(
                    df_to_predict,
                    model,
                    model_preprocessor,
                )

            if error:
                st.error(error)
                st.warning("Please ensure your CSV includes the required features. You can download the template from the sidebar.")
            else:
                total_cust = len(results_df)
                churn_count = int((results_df['Prediction'] == 'Churn (At Risk)').sum())
                retain_count = total_cust - churn_count
                churn_rate = (churn_count / total_cust) * 100 if total_cust > 0 else 0
                has_probabilities = 'Churn Probability (%)' in results_df

                # KPI tiles
                tiles = [
                    ("Total customers", f"{total_cust:,}", "analyzed", "#818cf8"),
                    ("Predicted churn", f"{churn_count:,}", f"{churn_rate:.1f}% of customers", "#f87171"),
                    ("Predicted retained", f"{retain_count:,}", f"{100 - churn_rate:.1f}% of customers", "#34d399"),
                ]
                if has_probabilities:
                    avg_prob = results_df['Churn Probability (%)'].mean()
                    tiles.append(("Avg churn probability", f"{avg_prob:.1f}%", "across all customers", "#fbbf24"))

                for column, (label, value, sub, accent) in zip(st.columns(len(tiles)), tiles):
                    with column:
                        md(f"""
                        <div class="kpi" style="--accent:{accent}">
                            <div class="lbl">{label}</div>
                            <div class="val">{value}</div>
                            <div class="sub">{sub}</div>
                        </div>
                        """)

                # Risk distribution
                if has_probabilities:
                    counts = results_df['Risk Level'].value_counts()
                    segments, legend = "", ""
                    for lvl in ["High Risk", "Medium Risk", "Low Risk"]:
                        n = int(counts.get(lvl, 0))
                        pct = n / total_cust * 100 if total_cust else 0
                        segments += f'<div style="width:{pct:.2f}%;background:{RISK_COLORS[lvl]}"></div>'
                        legend += f'<span><i style="background:{RISK_COLORS[lvl]}"></i>{lvl}: <b style="color:#fff">{n:,}</b> ({pct:.1f}%)</span>'
                    md('<div style="height:16px"></div>')
                    md(f"""
                    <div class="panel">
                        <div class="sec" style="margin:0">Risk distribution</div>
                        <div class="stack">{segments}</div>
                        <div class="legend">{legend}</div>
                    </div>
                    """)

                md('<div style="height:16px"></div>')
                section("📋", "Prediction results")

                # Filters and Search
                f_col1, f_col2 = st.columns([2, 1])
                with f_col1:
                    search_query = st.text_input("🔍 Search by customer ID or name", "")
                with f_col2:
                    filter_status = st.selectbox("Filter status", ["All Customers", "Churn (At Risk)", "Retained (Active)"])

                display_df = results_df.copy()
                if has_probabilities:
                    display_df = display_df.sort_values('Churn Probability (%)', ascending=False)

                if search_query:
                    mask = (
                        display_df['CustomerID'].astype(str).str.contains(search_query, case=False, na=False) |
                        display_df['Name'].astype(str).str.contains(search_query, case=False, na=False)
                    )
                    display_df = display_df[mask]

                if filter_status != "All Customers":
                    display_df = display_df[display_df['Prediction'] == filter_status]

                # Highlight styling (dark-theme friendly)
                def style_prediction(value):
                    if 'Churn' in str(value):
                        return 'background-color: rgba(248,113,113,0.18); color: #fca5a5; font-weight: 600;'
                    return 'background-color: rgba(52,211,153,0.16); color: #6ee7b7; font-weight: 600;'

                styled_table = display_df.style.map(style_prediction, subset=['Prediction'])
                column_config = {}
                if has_probabilities:
                    column_config['Churn Probability (%)'] = st.column_config.ProgressColumn(
                        "Churn probability",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    )

                st.dataframe(
                    styled_table,
                    width="stretch",
                    hide_index=True,
                    column_config=column_config,
                    height=min(420, 35 * len(display_df) + 38),
                )
                st.caption(f"Showing {len(display_df):,} of {total_cust:,} customers, highest risk first.")

                # Download Buttons
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    res_csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download results table",
                        data=res_csv,
                        file_name="churn_predictions_summary.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                with d_col2:
                    full_csv = full_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download full dataset with predictions",
                        data=full_csv,
                        file_name="churn_predictions_full.csv",
                        mime="text/csv",
                        width="stretch"
                    )

# ---------------------------------------------------------
# MODE 2: Single Customer Prediction
# ---------------------------------------------------------
elif app_mode == "👤 Single Customer Prediction":
    md("""
    <div class="page-title">👤 Single customer prediction</div>
    <div class="page-sub">Enter a customer's details to see their churn probability and the factors behind it.</div>
    """)

    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.form("single_customer_form", border=False):
            section(1, "Customer identification")
            id_col, name_col = st.columns(2)
            with id_col:
                cust_id = st.text_input("Customer ID", value="CUST-2024")
            with name_col:
                cust_name = st.text_input("Customer Name", value="Ahmed Mansour")

            section(2, "Demographics &amp; subscription")
            d1, d2 = st.columns(2)
            with d1:
                age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
            with d2:
                gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
            sub_type = st.radio("Subscription type", ["Basic", "Standard", "Premium"], index=1, horizontal=True)
            contract_length = st.radio("Contract length", ["Monthly", "Quarterly", "Annual"], index=2, horizontal=True)

            section(3, "Account activity &amp; support")
            a1, a2, a3 = st.columns(3)
            with a1:
                tenure = st.number_input("Tenure (months)", min_value=1, max_value=60, value=24, step=1)
                payment_delay = st.number_input("Payment delay (days)", min_value=0, max_value=30, value=3, step=1)
            with a2:
                usage_freq = st.number_input("Usage frequency (times/month)", min_value=1, max_value=30, value=18, step=1)
                last_interaction = st.number_input("Last interaction (days ago)", min_value=1, max_value=30, value=7, step=1)
            with a3:
                support_calls = st.number_input("Support calls", min_value=0, max_value=10, value=2, step=1)
                total_spend = st.number_input("Total spend ($)", min_value=100.0, max_value=1000.0, value=750.0, step=10.0)

            submit_btn = st.form_submit_button("⚡ Predict churn risk", width="stretch", type="primary")

    if submit_btn:
        if model is None:
            st.error(model_error)
        else:
            input_row = pd.DataFrame([{
                'Age': age,
                'Gender': gender,
                'Tenure': tenure,
                'Usage Frequency': usage_freq,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Subscription Type': sub_type,
                'Contract Length': contract_length,
                'Total Spend': total_spend,
                'Last Interaction': last_interaction
            }])

            with st.spinner("Calculating churn risk..."):
                model_features = prepare_model_features(input_row, model_preprocessor)
                pred = model.predict(model_features)[0]
                probabilities = get_churn_probabilities(model, model_features)

            st.session_state["single_result"] = {
                "name": cust_name,
                "id": cust_id,
                "pred": int(pred),
                "prob": float(probabilities[0] * 100) if probabilities is not None else None,
                "model": selected_model_name,
                "insights": build_insights(age, tenure, usage_freq, support_calls, payment_delay, contract_length),
                "inputs": {
                    "Tenure": f"{tenure} mo",
                    "Plan": sub_type,
                    "Contract": contract_length,
                    "Spend": f"${total_spend:,.0f}",
                    "Support calls": support_calls,
                    "Payment delay": f"{payment_delay} d",
                },
            }

    with right:
        result = st.session_state.get("single_result")
        if result is None:
            md("""
            <div class="placeholder">
                <div class="ph-ico">🔮</div>
                <div class="ph-title">No prediction yet</div>
                <div class="ph-sub">Fill in the customer details and press <b>Predict churn risk</b> to see the result here.</div>
            </div>
            """)
        else:
            render_result(result)

# ---------------------------------------------------------
# MODE 3: Model & Insights
# ---------------------------------------------------------
elif app_mode == "📊 Model & Insights":
    md("""
    <div class="page-title">📊 Model &amp; insights</div>
    <div class="page-sub">What the selected model is, what it looks at, and what drives its predictions.</div>
    """)

    if model is None:
        st.error(model_error)
    else:
        named_steps = getattr(model, 'named_steps', {})
        classifier = named_steps.get('classifier', model)
        preprocessor = named_steps.get('preprocessor', model_preprocessor)
        supports_proba = callable(getattr(model, 'predict_proba', None))
        summary = load_dataset_summary()

        tiles = [
            ("Selected model", html.escape(selected_model_name), type(classifier).__name__, "#818cf8"),
            ("Input features", f"{len(EXPECTED_NUM_COLS) + len(EXPECTED_CAT_COLS)}", f"{len(EXPECTED_NUM_COLS)} numeric · {len(EXPECTED_CAT_COLS)} categorical", "#a78bfa"),
            ("Probabilities", "Available" if supports_proba else "Not supported", "class 1 = churn", "#34d399" if supports_proba else "#fbbf24"),
        ]
        if summary is not None:
            tiles.append(("Training data", f"{summary[0]:,}", f"{summary[1]:.1f}% churned", "#fbbf24"))

        for column, (label, value, sub, accent) in zip(st.columns(len(tiles)), tiles):
            with column:
                md(f"""
                <div class="kpi" style="--accent:{accent}">
                    <div class="lbl">{label}</div>
                    <div class="val" style="font-size:1.5rem">{value}</div>
                    <div class="sub">{sub}</div>
                </div>
                """)

        md('<div style="height:20px"></div>')
        section("🎯", "What drives the predictions")
        importance = get_feature_importance(model, model_preprocessor)
        if importance:
            top = importance[0][1] or 1.0
            rows = "".join(
                f'<div class="fi-row"><div class="nm">{html.escape(name)}</div>'
                f'<div class="tr"><div class="fl" style="width:{value / top * 100:.1f}%"></div></div>'
                f'<div class="pc">{value:.1f}%</div></div>'
                for name, value in importance
            )
            md(f'<div class="panel">{rows}</div>')
            st.caption("Relative feature importance reported by the model itself, not a verified performance metric.")
        else:
            st.info("This model does not expose feature importances.")

        md('<div style="height:12px"></div>')
        with st.expander("⚙️ Preprocessing details"):
            if preprocessor is not None:
                st.markdown("The saved pipeline applies its own preprocessing before prediction.")
                for transformer_name, transformer, columns in preprocessor.transformers_:
                    if transformer_name != 'remainder':
                        st.write(f"**{transformer_name}:** {', '.join(map(str, columns))}")
                        st.caption(f"Preprocessing: {transformer}")
            else:
                st.info("This saved model does not expose a preprocessing pipeline.")
            st.write("**Supported classes:**", list(getattr(model, 'classes_', [])))

        st.caption("No performance metrics are shown because no verified evaluation report is loaded by this app.")
        st.markdown("---")
        st.info("Created for the **NTI Machine Learning Training Program**.")
