# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon="💰",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

with open("models/adaboost_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("models/feature_columns.pkl", "rb") as file:
    feature_columns = pickle.load(file)

# ==========================================
# TITLE
# ==========================================

st.title("💰 Insurance Cost Prediction")

st.markdown("""
Predict medical insurance charges using an
**AdaBoost Regressor** model.
""")

st.divider()

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("Enter Customer Details")

age = st.sidebar.slider(
    "Age",
    18,
    100,
    30
)

sex = st.sidebar.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.sidebar.slider(
    "BMI",
    10.0,
    50.0,
    25.0
)

children = st.sidebar.slider(
    "Children",
    0,
    10,
    0
)

smoker = st.sidebar.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.sidebar.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

# ==========================================
# INPUT DATA
# ==========================================

input_dict = {
    "age": age,
    "bmi": bmi,
    "children": children,
    "sex_male": 1 if sex == "male" else 0,
    "smoker_yes": 1 if smoker == "yes" else 0,
    "region_northwest":
        1 if region == "northwest" else 0,
    "region_southeast":
        1 if region == "southeast" else 0,
    "region_southwest":
        1 if region == "southwest" else 0
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

    st.subheader(" Customer Information")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button("Predict Insurance Cost"):

        prediction = model.predict(input_df)[0]

        st.divider()

        # ==================================
        # PREDICTION RESULT
        # ==================================

        st.subheader(" Predicted Insurance Charges")

        st.success(
            f"Estimated Insurance Cost: ₹ {prediction:,.2f}"
        )

        # ==================================
        # RISK CATEGORY
        # ==================================

        if prediction < 10000:
            st.info("Category: Low Insurance Cost")

        elif prediction < 30000:
            st.info("Category: Medium Insurance Cost")

        else:
            st.warning("Category: High Insurance Cost")

        # ==================================
        # FEATURE IMPORTANCE
        # ==================================

        st.divider()

        st.subheader(" Feature Importance")

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

    st.subheader(" Model Details")

    st.metric(
        label="Algorithm",
        value="AdaBoost"
    )

    st.metric(
        label="Problem Type",
        value="Regression"
    )

    st.metric(
        label="Dataset",
        value="Insurance Cost"
    )

    st.metric(
        label="Features",
        value=len(feature_columns)
    )

    st.divider()

    st.subheader(" About AdaBoost")

    st.write("""
    ✔ Boosting Algorithm
    
    ✔ Sequential Learning
    
    ✔ Improves Weak Learners
    
    ✔ Reduces Prediction Errors
    
    ✔ Good for Regression Tasks
    """)