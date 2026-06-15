import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE ---------------- #
st.set_page_config(page_title="Loan Predictor", page_icon="💰")

st.title("💰 Loan Approval Prediction")
st.write("Predict whether your loan will be approved")

st.markdown("---")

# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

# ---------------- CLEAN DATA ---------------- #

# Fix Dependents (IMPORTANT 🔥)
df["Dependents"] = df["Dependents"].replace("3+", 3)
df["Dependents"] = df["Dependents"].fillna(0).astype(int)

# Fill missing values
for col in ["Gender", "Married", "Self_Employed"]:
    df[col] = df[col].fillna(df[col].mode()[0])

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())
df["Credit_History"] = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

# Target
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Drop ID
df = df.drop("Loan_ID", axis=1)

# ---------------- FEATURES ---------------- #
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Encode
X = pd.get_dummies(X, drop_first=True)

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ---------------- UI ---------------- #

st.subheader("👤 Personal Info")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Married = st.selectbox("Married", ["Yes", "No"])
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"])

with col2:
    Dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
    Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

st.subheader("💵 Financial Info")

col3, col4 = st.columns(2)

with col3:
    ApplicantIncome = st.number_input("Applicant Income", min_value=0.0)
    LoanAmount = st.number_input("Loan Amount", min_value=0.0)

with col4:
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
    Loan_Amount_Term = st.number_input("Loan Term", min_value=0.0)

Credit_History = st.selectbox("Credit History", [1.0, 0.0])

st.markdown("---")

# ---------------- PREDICT ---------------- #

if st.button("🚀 Predict"):

    input_df = pd.DataFrame({
        "Gender": [Gender],
        "Married": [Married],
        "Dependents": [int(Dependents)],  # 🔥 FIX
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
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Ensure numeric
    input_df = input_df.astype(float)

    # Predict
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")

    st.info(f"📊 Approval Probability: {round(prob*100, 2)}%")