# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

with open("models/adaboost_classifier.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# ==========================================
# TITLE
# ==========================================

st.title("👨‍💼 Employee Attrition Prediction")

st.markdown("""
Predict whether an employee is likely to leave
the company using an **AdaBoost Classifier**.
""")

st.divider()

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Employee Details")

age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

monthly_income = st.sidebar.slider(
    "Monthly Income",
    1000,
    50000,
    10000
)

distance_from_home = st.sidebar.slider(
    "Distance From Home",
    1,
    50,
    5
)

years_at_company = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    5
)

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    1,
    4,
    3
)

environment_satisfaction = st.sidebar.slider(
    "Environment Satisfaction",
    1,
    4,
    3
)

work_life_balance = st.sidebar.slider(
    "Work Life Balance",
    1,
    4,
    3
)

job_level = st.sidebar.slider(
    "Job Level",
    1,
    5,
    2
)

overtime = st.sidebar.selectbox(
    "OverTime",
    ["Yes", "No"]
)

business_travel = st.sidebar.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)

marital_status = st.sidebar.selectbox(
    "Marital Status",
    [
        "Single",
        "Married",
        "Divorced"
    ]
)

# ==========================================
# INPUT DATA
# ==========================================

input_dict = {
    "Age": age,
    "MonthlyIncome": monthly_income,
    "DistanceFromHome": distance_from_home,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "EnvironmentSatisfaction": environment_satisfaction,
    "WorkLifeBalance": work_life_balance,
    "JobLevel": job_level,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "BusinessTravel_Travel_Frequently":
        1 if business_travel == "Travel_Frequently" else 0,
    "BusinessTravel_Travel_Rarely":
        1 if business_travel == "Travel_Rarely" else 0,
    "MaritalStatus_Married":
        1 if marital_status == "Married" else 0,
    "MaritalStatus_Single":
        1 if marital_status == "Single" else 0
}

input_df = pd.DataFrame([input_dict])

# Add missing columns
for col in feature_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Correct column order
input_df = input_df[feature_columns]

# ==========================================
# MAIN LAYOUT
# ==========================================

col1, col2 = st.columns([2, 1])

# ==========================================
# LEFT SECTION
# ==========================================

with col1:

    st.subheader("📋 Employee Information")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button("Predict Attrition"):

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0][1]

        st.divider()

        # ==================================
        # PREDICTION RESULT
        # ==================================

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                f"⚠ Employee Likely to Leave\n\n"
                f"Probability: {probability:.2%}"
            )

        else:

            st.success(
                f"✅ Employee Likely to Stay\n\n"
                f"Probability: {(1 - probability):.2%}"
            )

        # ==================================
        # FEATURE IMPORTANCE
        # ==================================

        st.divider()

        st.subheader("📈 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            importance_df["Feature"],
            importance_df["Importance"]
        )

        ax.set_xlabel("Importance")

        ax.set_title("Top 10 Important Features")

        plt.gca().invert_yaxis()

        st.pyplot(fig)

# ==========================================
# RIGHT SECTION
# ==========================================

with col2:

    st.subheader("🤖 Model Details")

    st.metric(
        label="Algorithm",
        value="AdaBoost"
    )

    st.metric(
        label="Problem Type",
        value="Classification"
    )

    st.metric(
        label="Dataset",
        value="Employee Attrition"
    )

    st.metric(
        label="Features",
        value=len(feature_columns)
    )

    st.divider()

    st.subheader("📌 About AdaBoost")

    st.write("""
    ✔ Boosting Algorithm
    
    ✔ Sequential Learning
    
    ✔ Improves Weak Learners
    
    ✔ High Classification Accuracy
    
    ✔ Reduces Prediction Errors
    """)