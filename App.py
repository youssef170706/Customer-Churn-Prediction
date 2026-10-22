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
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Keep every slider's track and thumb left-to-right, even when the
       surrounding page or browser uses a right-to-left text direction. */
    div[data-testid="stSlider"],
    div[data-testid="stSlider"] > div,
    div[data-testid="stSlider"] [data-baseweb="slider"],
    div[data-testid="stSlider"] [data-baseweb="slider"] > div {
        direction: ltr !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        direction: ltr !important;
    }

    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 14px;
        color: white;
        margin-bottom: 24px;
        border-left: 6px solid #6366f1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Result card styling */
    .result-card-churn {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
    }
    .result-card-retain {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    /* Custom metric tiles */
    .metric-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Badges */
    .badge-churn {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-retain {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

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

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/analytics.png", width=110)
    st.title("Navigation")
    
    app_mode = st.radio(
        "Choose Mode:",
        ["📁 Batch CSV Prediction", "👤 Single Customer Prediction", "📊 Model & Insights"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Prediction Model")
    selected_model_name = st.selectbox(
        "Choose a model:",
        list(MODEL_PATHS),
    )
    model, model_preprocessor, model_error = loaded_models[selected_model_name]

    st.markdown("### ⚙️ System Status")
    for model_name, (loaded_model, _, load_error) in loaded_models.items():
        if loaded_model is not None:
            st.success(f"{model_name} loaded")
        else:
            st.error(f"{model_name} unavailable: {load_error}")
        
    st.markdown("---")
    st.markdown("### 💡 Quick Download")
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
        use_container_width=True
    )
    
    st.markdown("---")
    st.caption("NTI Machine Learning Project\n\nDeveloped with Streamlit & Scikit-Learn")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🔮 Customer Churn Prediction System</h1>
    <p>Predict whether a customer is likely to stay or churn using the selected trained model.</p>
</div>
""", unsafe_allow_html=True)

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


# ---------------------------------------------------------
# MODE 1: Batch CSV Prediction
# ---------------------------------------------------------
if app_mode == "📁 Batch CSV Prediction":
    st.subheader("📁 Upload CSV for Batch Customer Churn Prediction")
    st.markdown("Upload a CSV file containing customer details. The app will return each **CustomerID**, **Name**, and **Prediction**, plus churn probabilities when supported by the selected model.")

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        use_demo = st.button("🧪 Try with Demo Data", help="Load test data with Ahmed, Sara, Omar, etc.", use_container_width=True)

    df_to_predict = None

    if use_demo:
        if SAMPLE_CSV_PATH.is_file():
            df_to_predict = pd.read_csv(SAMPLE_CSV_PATH)
            st.info("Loaded demo test dataset with 8 sample customers.")
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

    if df_to_predict is not None:
        st.markdown("---")
        with st.expander("👀 Preview Uploaded Data (First 5 Rows)", expanded=False):
            st.dataframe(df_to_predict.head(), use_container_width=True)

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
                churn_count = (results_df['Prediction'] == 'Churn (At Risk)').sum()
                retain_count = total_cust - churn_count
                churn_rate = (churn_count / total_cust) * 100 if total_cust > 0 else 0
                has_probabilities = 'Churn Probability (%)' in results_df
                metric_columns = st.columns(4 if has_probabilities else 3)

                # Metric Cards
                with metric_columns[0]:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-lbl">Total Customers</div>
                        <div class="metric-val">{total_cust:,}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_columns[1]:
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: #ef4444;">
                        <div class="metric-lbl" style="color: #ef4444;">Predicted Churn</div>
                        <div class="metric-val" style="color: #ef4444;">{churn_count:,} ({churn_rate:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with metric_columns[2]:
                    st.markdown(f"""
                    <div class="metric-box" style="border-color: #10b981;">
                        <div class="metric-lbl" style="color: #10b981;">Predicted Retained</div>
                        <div class="metric-val" style="color: #10b981;">{retain_count:,} ({100-churn_rate:.1f}%)</div>
                    </div>
                    """, unsafe_allow_html=True)
                if has_probabilities:
                    avg_prob = results_df['Churn Probability (%)'].mean()
                    with metric_columns[3]:
                        st.markdown(f"""
                        <div class="metric-box" style="border-color: #6366f1;">
                            <div class="metric-lbl" style="color: #6366f1;">Avg Churn Probability</div>
                            <div class="metric-val" style="color: #6366f1;">{avg_prob:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Filters and Search
                f_col1, f_col2 = st.columns([2, 1])
                with f_col1:
                    search_query = st.text_input("🔍 Search by Customer ID or Name", "")
                with f_col2:
                    filter_status = st.selectbox("Filter Status", ["All Customers", "Churn (At Risk)", "Retained (Active)"])

                display_df = results_df.copy()
                if search_query:
                    mask = (
                        display_df['CustomerID'].astype(str).str.contains(search_query, case=False, na=False) |
                        display_df['Name'].astype(str).str.contains(search_query, case=False, na=False)
                    )
                    display_df = display_df[mask]

                if filter_status != "All Customers":
                    display_df = display_df[display_df['Prediction'] == filter_status]

                st.markdown("### 📋 Prediction Results (ID, Name, and Prediction)")
                
                # Highlight styling function
                def style_predictions(row):
                    if 'Churn' in str(row['Prediction']):
                        return ['background-color: #fee2e2; color: #991b1b; font-weight: bold;' if col == 'Prediction' else '' for col in row.index]
                    else:
                        return ['background-color: #dcfce7; color: #166534; font-weight: bold;' if col == 'Prediction' else '' for col in row.index]

                styled_table = display_df.style.apply(style_predictions, axis=1)
                st.dataframe(styled_table, use_container_width=True, height=min(400, 35 * len(display_df) + 38))

                # Download Buttons
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    res_csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results Table (ID, Name, Prediction)",
                        data=res_csv,
                        file_name="churn_predictions_summary.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with d_col2:
                    full_csv = full_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Dataset With Predictions",
                        data=full_csv,
                        file_name="churn_predictions_full.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# ---------------------------------------------------------
# MODE 2: Single Customer Prediction
# ---------------------------------------------------------
elif app_mode == "👤 Single Customer Prediction":
    st.subheader("👤 Predict Churn for a Single Customer")
    st.markdown("Enter customer information below to predict churn and view a probability when the selected model supports it.")

    with st.form("single_customer_form"):
        st.markdown("#### 1. Customer Identification")
        id_col, name_col = st.columns(2)
        with id_col:
            cust_id = st.text_input("Customer ID", value="CUST-2024")
        with name_col:
            cust_name = st.text_input("Customer Name", value="Ahmed Mansour")

        st.markdown("#### 2. Demographics & Subscription")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            age = st.number_input("Age", min_value=18, max_value=65, value=35, step=1)
        with d2:
            gender = st.selectbox("Gender", ["Female", "Male"])
        with d3:
            sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], index=1)
        with d4:
            contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"], index=2)

        st.markdown("#### 3. Account Activity & Support")
        a1, a2, a3 = st.columns(3)
        with a1:
            tenure = st.number_input("Tenure (Months)", min_value=1, max_value=60, value=24, step=1)
            usage_freq = st.number_input("Usage Frequency (Times/Month)", min_value=1, max_value=30, value=18, step=1)
        with a2:
            support_calls = st.number_input("Support Calls", min_value=0, max_value=10, value=2, step=1)
            payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=30, value=3, step=1)
        with a3:
            total_spend = st.number_input("Total Spend ($)", min_value=100.0, max_value=1000.0, value=750.0, step=10.0)
            last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=1, max_value=30, value=7, step=1)

        submit_btn = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True)

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
                if probabilities is not None:
                    prob_churn = probabilities[0] * 100
                    prob_stay = 100 - prob_churn

            st.markdown("---")
            st.markdown("### 🎯 Prediction Verdict")
            probability_details = (
                f' has a <strong>{prob_churn:.1f}%</strong> probability of churning.'
                if probabilities is not None
                else ' is predicted to churn.'
            )

            if pred == 1:
                st.markdown(f"""
                <div class="result-card-churn">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="color: #b91c1c; margin: 0;">🚨 High Risk of Churn</h2>
                        <span class="badge-churn">CHURN: YES</span>
                    </div>
                    <p style="color: #7f1d1d; font-size: 1.1rem; margin-top: 10px;">
                        Customer <strong>{html.escape(cust_name)}</strong> (ID: <code>{html.escape(cust_id)}</code>){probability_details}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                probability_details = (
                    f' has a <strong>{prob_stay:.1f}%</strong> probability of retention ({prob_churn:.1f}% churn risk).'
                    if probabilities is not None
                    else ' is predicted to remain a customer.'
                )
                st.markdown(f"""
                <div class="result-card-retain">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h2 style="color: #047857; margin: 0;">✅ Customer Likely to Stay</h2>
                        <span class="badge-retain">CHURN: NO</span>
                    </div>
                    <p style="color: #064e3b; font-size: 1.1rem; margin-top: 10px;">
                        Customer <strong>{html.escape(cust_name)}</strong> (ID: <code>{html.escape(cust_id)}</code>){probability_details}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Progress Bars
            if probabilities is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    st.markdown(f"**🔴 Churn Probability: `{prob_churn:.1f}%`**")
                    st.progress(float(prob_churn / 100))
                with p_col2:
                    st.markdown(f"**🟢 Retention Probability: `{prob_stay:.1f}%`**")
                    st.progress(float(prob_stay / 100))

            # Explainable insights
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 💡 Risk Factors & Insights")
            insights = []
            if payment_delay >= 15:
                insights.append(f"⚠️ **High Payment Delay ({payment_delay} days):** Extended payment delays strongly correlate with churn.")
            if support_calls >= 5:
                insights.append(f"⚠️ **Frequent Support Calls ({support_calls} calls):** High call volume indicates customer frustration.")
            if contract_length == "Monthly":
                insights.append("⚠️ **Monthly Contract:** Month-to-month contracts have significantly higher turnover compared to Annual contracts.")
            if usage_freq <= 5:
                insights.append(f"⚠️ **Low Usage Frequency ({usage_freq} times/mo):** Inactive customers are at higher risk of leaving.")
            if tenure <= 6:
                insights.append(f"ℹ️ **Short Tenure ({tenure} months):** New customers require onboarding care to prevent early churn.")

            if insights:
                for ins in insights:
                    st.write(ins)
            else:
                st.success("✨ Good customer health metrics: low payment delay, reasonable support call volume, and steady activity.")


# ---------------------------------------------------------
# MODE 3: Model & Insights
# ---------------------------------------------------------
elif app_mode == "📊 Model & Insights":
    st.subheader("📊 Model Architecture & Training Insights")
    if model is None:
        st.error(model_error)
    else:
        named_steps = getattr(model, 'named_steps', {})
        classifier = named_steps.get('classifier', model)
        st.markdown(f"Selected model: **{selected_model_name}** (`{type(classifier).__name__}`)")

        preprocessor = named_steps.get('preprocessor', model_preprocessor)
        if preprocessor is not None:
            st.markdown("The saved pipeline applies its own preprocessing before prediction.")
            for transformer_name, transformer, columns in preprocessor.transformers_:
                if transformer_name != 'remainder':
                    st.write(f"**{transformer_name}:** {', '.join(map(str, columns))}")
                    st.caption(f"Preprocessing: {transformer}")
        elif model_preprocessor is None:
            st.info("This saved model does not expose a preprocessing pipeline.")

        st.write("**Supported classes:**", list(getattr(model, 'classes_', [])))
        st.write(
            "**Churn probabilities:**",
            "available" if callable(getattr(model, 'predict_proba', None)) else "not supported",
        )
        st.caption("No performance metrics are shown because no verified evaluation report is loaded by this app.")

    st.markdown("---")
    st.info("Created for the **NTI Machine Learning Training Program**.")
