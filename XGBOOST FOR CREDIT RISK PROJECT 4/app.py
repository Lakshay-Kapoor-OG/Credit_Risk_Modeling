import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="centered"
)

# ==========================
# Load Model & Encoders
# ==========================
@st.cache_resource
def load_artifacts():
    model = joblib.load("extra_trees_credit_model.pkl")

    encoders = {
        "Sex": joblib.load("Sex_encoder.pkl"),
        "Housing": joblib.load("Housing_encoder.pkl"),
        "Saving accounts": joblib.load("Saving accounts_encoder.pkl"),
        "Checking account": joblib.load("Checking account_encoder.pkl")
    }

    return model, encoders


model, encoders = load_artifacts()

# ==========================
# Title
# ==========================
st.title("💳 Credit Risk Prediction")
st.write(
    "Enter the applicant's information below to predict whether the credit risk is **Good** or **Bad**."
)

# ==========================
# User Inputs
# ==========================
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

job = st.selectbox(
    "Job",
    [0, 1, 2, 3],
    help="""
0 = Unemployed / Unskilled (Non-resident)

1 = Unskilled (Resident)

2 = Skilled

3 = Highly Skilled
"""
)

housing = st.selectbox(
    "Housing",
    ["own", "rent", "free"]
)

saving_accounts = st.selectbox(
    "Saving Accounts",
    ["little", "moderate", "rich", "quite rich"]
)

checking_account = st.selectbox(
    "Checking Account",
    ["little", "moderate", "rich", "quite rich"]
)

credit_amount = st.number_input(
    "Credit Amount",
    min_value=0,
    value=1000
)

duration = st.number_input(
    "Duration (Months)",
    min_value=1,
    value=12
)

# ==========================
# Prediction
# ==========================
if st.button("Predict Risk"):

    try:

        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.success("✅ Predicted Credit Risk: GOOD")
        else:
            st.error("❌ Predicted Credit Risk: BAD")

    except Exception as e:
        st.error(f"An error occurred:\n{e}")