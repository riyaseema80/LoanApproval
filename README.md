# 💰 Loan Approval Prediction App

A Machine Learning web application built using **Streamlit** that predicts whether a loan will be approved based on user inputs.

---

## 🚀 Demo
👉 Live App: *([Streamlit link here](https://loanapproval-4nuloowz2s94stjoqdvjfr.streamlit.app/))*

---

## 📌 Project Overview

This project uses **Logistic Regression** to solve a classification problem.

### 🎯 Goal:
Predict whether a loan application will be:

- ✅ Approved  
- ❌ Not Approved  

based on applicant details.

---

## 🧠 Features Used

- Gender  
- Married  
- Dependents  
- Education  
- Self Employed  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Term  
- Credit History  
- Property Area  

---

## 🛠️ Tech Stack

- Python 🐍  
- Pandas 📊  
- Scikit-learn 🤖  
- Streamlit 🌐  

---

## ⚙️ How It Works

1. Load dataset (CSV)
2. Handle missing values
3. Encode categorical data
4. Train Logistic Regression model
5. Take user input from UI
6. Predict loan status
7. Display result with probability

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/LoanApproval.git
cd LoanApproval
pip install -r requirements.txt
streamlit run app.py
