# -------------------- PATH FIX (IMPORTANT) --------------------
import sys
from pathlib import Path

# Add project root to PYTHONPATH (robust for Streamlit & Windows)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from src.evaluate import assign_risk_bucket

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Credit Risk Assessment",
    layout="centered"
)

MODEL_PATH = "models/xgboost_model.pkl"

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------- UI --------------------
st.title("💳 AI-Driven Credit Risk Assessment")
st.write("Enter applicant details to predict loan default risk.")

# -------------------- INPUT FORM --------------------
with st.form("credit_form"):
    loan_amnt = st.number_input("Loan Amount", min_value=1000, value=10000)
    annual_inc = st.number_input("Annual Income", min_value=1000, value=60000)
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
    installment = st.number_input("Monthly Installment", min_value=100, value=300)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 15.0)
    credit_length = st.slider("Credit History Length (Years)", 0, 40, 10)
    revol_bal = st.number_input("Revolving Balance", min_value=0, value=5000)
    delinq_2yrs = st.selectbox("Delinquencies in Last 2 Years", [0, 1, 2, 3])

    submit = st.form_submit_button("Predict Risk")

# -------------------- FEATURE ENGINEERING --------------------
def build_input_dataframe():
    income_to_loan = annual_inc / (loan_amnt + 1)
    installment_to_income = installment / (annual_inc / 12 + 1)
    credit_utilization = revol_bal / (loan_amnt + 1)
    has_delinquency = 1 if delinq_2yrs > 0 else 0

    data = {
        "loan_amnt": loan_amnt,
        "term": " 36 months",
        "int_rate": int_rate,
        "installment": installment,
        "grade": "B",
        "sub_grade": "B2",
        "emp_length": "5 years",
        "home_ownership": "RENT",
        "annual_inc": annual_inc,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": 1,
        "open_acc": 5,
        "pub_rec": 0,
        "revol_bal": revol_bal,
        "revol_util": 30.0,
        "total_acc": 20,
        "credit_length": credit_length,
        "income_to_loan": income_to_loan,
        "installment_to_income": installment_to_income,
        "credit_utilization": credit_utilization,
        "has_delinquency": has_delinquency,
        "credit_length_bin": "Medium",
        "int_rate_bin": "Medium"
    }

    return pd.DataFrame([data])

# -------------------- PREDICTION --------------------
if submit:
    input_df = build_input_dataframe()

    prob_default = model.predict_proba(input_df)[0][1]
    risk = assign_risk_bucket(prob_default)

    st.subheader("📊 Prediction Results")
    st.metric("Default Probability", f"{prob_default:.2%}")
    st.metric("Risk Category", risk)

    st.progress(int(prob_default * 100))

    st.caption("⚠️ This prediction is AI-assisted and for demonstration purposes only.")
