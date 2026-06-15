import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

# ---------------- TITLE ---------------- #
st.title("💰 Loan Approval Prediction")
st.write("Predict whether your loan will be approved")

st.markdown("---")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# ---------------- DATA PREPROCESSING ---------------- #

# Fill missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# 🔥 Fix Dependents safely
df["Dependents"] = df["Dependents"].replace("3+", "3")
df["Dependents"] = pd.to_numeric(df["Dependents"], errors='coerce')
df["Dependents"] = df["Dependents"].fillna(0)

# Drop ID
df.drop("Loan_ID", axis=1, inplace=True)

# Convert target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# One-hot encoding
X = pd.get_dummies(df.drop("Loan_Status", axis=1))
y = df["Loan_Status"]

# Ensure clean numeric
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X, y)

# ---------------- UI ---------------- #

st.subheader("Enter Details")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    ApplicantIncome = st.number_input("Applicant Income", 0.0)
    LoanAmount = st.number_input("Loan Amount", 0.0)

with col4:
    CoapplicantIncome = st.number_input("Coapplicant Income", 0.0)
    Loan_Amount_Term = st.number_input("Loan Term", 0.0)

Credit_History = st.selectbox("Credit History", [1.0, 0.0])

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Predict"):

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

    # 🔥 SAME FIX FOR INPUT
    input_df["Dependents"] = input_df["Dependents"].replace("3+", "3")
    input_df["Dependents"] = pd.to_numeric(input_df["Dependents"], errors='coerce')
    input_df["Dependents"] = input_df["Dependents"].fillna(0)

    # Encode input
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Final clean
    input_df = input_df.astype(float).fillna(0)

    # Prediction
    result = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    # Output
    st.markdown("---")

    if result[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"📊 Probability: {round(prob * 100, 2)}%")
