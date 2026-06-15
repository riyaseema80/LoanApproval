import streamlit as st
import pandas as pd
import joblib

# Load model + columns
model = joblib.load("loan_model.pkl")
columns = joblib.load("columns.pkl")

st.title("💰 Loan Approval Prediction")

# Inputs
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

ApplicantIncome = st.number_input("Applicant Income", 0.0)
CoapplicantIncome = st.number_input("Coapplicant Income", 0.0)
LoanAmount = st.number_input("Loan Amount", 0.0)
Loan_Amount_Term = st.number_input("Loan Term", 0.0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])

if st.button("Predict"):

    # Create input
    input_df = pd.DataFrame({
        "Gender": [Gender],
        "Married": [Married],
        "Dependents": [Dependents],
        "Education": [Education],
        "Self_Employed": [Self_Employed],
        "ApplicantIncome": [ApplicantIncome],
        "CoapplicantIncome": [CoapplicantIncome],
        "LoanAmount": [LoanAmount],
        "Loan_Amount_Term": [Loan_Amount_Term],
        "Credit_History": [Credit_History],
        "Property_Area": [Property_Area]
    })

    # 🔥 Same encoding as training
    input_df = pd.get_dummies(input_df)

    # Match columns
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # Predict
    result = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"Probability: {round(prob*100, 2)}%")
