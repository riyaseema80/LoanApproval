import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Loan Predictor", page_icon="💰", layout="centered")

# ---------------- TITLE ---------------- #
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💰 Loan Approval Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether your loan will be approved</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD DATA ---------------- #
@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

    # Fill missing values
    df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
    df["Married"].fillna(df["Married"].mode()[0], inplace=True)
    df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
    df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)

    df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
    df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)

    df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    df = df.drop("Loan_ID", axis=1)

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    X = pd.get_dummies(X, drop_first=True)

    return X, y

X, y = load_data()

# ---------------- TRAIN MODEL ---------------- #
@st.cache_resource
def train_model(X, y):
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model

model = train_model(X, y)

# ---------------- UI ---------------- #

st.subheader("👤 Personal Information")

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

st.subheader("💵 Financial Information")

col3, col4 = st.columns(2)

with col3:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0, value=5000.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0, value=150.0)

with col4:
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term", min_value=0.0, value=360.0)

Credit_History = st.selectbox("Credit History", [1.0, 0.0])

st.markdown("---")

# ---------------- PREDICTION ---------------- #

if st.button("🚀 Predict Loan Status"):

    input_data = pd.DataFrame({
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

    # Encode input
    input_data = pd.get_dummies(input_data)

    # Align columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # 🔥 FINAL SAFETY
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_data = input_data.fillna(0)

    # Prediction
    result = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if result == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"📊 Approval Probability: {round(probability*100, 2)}%")