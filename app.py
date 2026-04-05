import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="Churn Intelligence System", layout="centered")

# ---------------- TITLE ---------------- #
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚀 Customer Churn Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Predict customer churn with AI-powered insights</h4>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("churn_model.pkl")

# ---------------- INPUTS ---------------- #
st.subheader("📋 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

with col2:
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

# ---------------- MORE INPUTS ---------------- #
st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

with col4:
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

# ---------------- NUMERICAL ---------------- #
st.markdown("---")

tenure = st.slider("Tenure (Months)", 0, 72, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# ---------------- FEATURE ENGINEERING ---------------- #
AvgCharges = TotalCharges / (tenure + 1)

ServiceCount = sum([
    OnlineSecurity == "Yes",
    OnlineBackup == "Yes",
    DeviceProtection == "Yes",
    TechSupport == "Yes",
    StreamingTV == "Yes",
    StreamingMovies == "Yes"
])

TotalSpendIntensity = MonthlyCharges * tenure

# ---------------- PREDICTION ---------------- #
st.markdown("### 🔍 Prediction")

if st.button("🚀 Predict Churn"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "AvgCharges": AvgCharges,
        "ServiceCount": ServiceCount,
        "TotalSpendIntensity": TotalSpendIntensity
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Churn\n\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ Customer Likely to Stay\n\nProbability: {probability:.2f}")

    st.progress(int(probability * 100))

    # ---------------- BUSINESS INSIGHTS ---------------- #
    st.markdown("### 💡 Business Insight")

    if prediction == 1:
        st.warning("""
        🔴 High-risk customer detected

        Recommended Actions:
        - Offer discounts or retention offers  
        - Improve customer support  
        - Provide personalized engagement  
        """)
    else:
        st.info("""
        🟢 Customer likely to stay

        Opportunities:
        - Upsell premium plans  
        - Improve long-term engagement  
        """)

    # ---------------- SHAP EXPLAINABILITY ---------------- #
st.markdown("### 📊 Model Explanation (SHAP)")

try:
    # Get final model from pipeline automatically
    model_only = model.named_steps[list(model.named_steps.keys())[-1]]
    
    explainer = shap.Explainer(model_only)
    shap_values = explainer(input_data)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

except Exception as e:
    st.info(f"SHAP not available for this model.")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This application predicts customer churn using a machine learning pipeline.\n\n"
    "Includes feature engineering, SMOTE handling, model tuning, and explainability."
)
