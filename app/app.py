import streamlit as st
import pandas as pd

from utils import predict_risk

st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="💳",
    layout="wide"
)

# Professional CSS

st.markdown(
    """
<style>

.main{
background-color:#f7f9fc;
}

h1{
color:#003366;
}

.stButton>button{
background:#003366;
color:white;
border-radius:8px;
height:45px;
width:100%;
font-size:18px;
}

.metric-container{
padding:15px;
border-radius:10px;
background:white;
}

</style>
""",
    unsafe_allow_html=True,
)

#Sidebar

st.sidebar.title("🏦 Credit Risk Assessment")

st.sidebar.markdown("---")

st.sidebar.write("### Project Information")

st.sidebar.write("""
Model : XGBoost

Dataset : German Credit Dataset

Objective :
Predict whether a customer is a good or bad credit risk.
""")

st.sidebar.markdown("---")

st.sidebar.write("Developed using")

st.sidebar.write("""
✔ Streamlit

✔ XGBoost

✔ Scikit-Learn

✔ Python
""")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Technologies")

st.sidebar.markdown("""
- Python
- Streamlit
- XGBoost
- Scikit-Learn
- Pandas
- NumPy
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This application predicts whether a loan applicant is a Good or Bad credit risk using an XGBoost classifier trained on the German Credit dataset."
)

#MAIN TITLE

st.title("🏦 Banking Credit Risk Assessment Dashboard")

st.write(
"""
This application predicts the creditworthiness of a loan applicant
using a Machine Learning model trained on the German Credit dataset.
"""
)

left, right = st.columns(2)

#User Input Form

with left:

    st.subheader("Customer Information")

    age = st.number_input(
        "Age",
        18,
        100,
        30
    )

    sex = st.selectbox(
        "Gender",
        ["male", "female"]
    )

    job = st.selectbox(
        "Job Level",
        [0, 1, 2, 3]
    )

    housing = st.selectbox(
        "Housing",
        ["own", "rent", "free"]
    )

    saving = st.selectbox(
        "Saving Account",
        ["little", "moderate", "quite rich", "rich"]
    )

    checking = st.selectbox(
    "Checking Account",
    [
        "little",
        "moderate",
        "rich",
        "unknown"
    ]
)
    credit_amount = st.number_input(
        "Credit Amount",
        100,
        30000,
        5000
    )

    duration = st.slider(
        "Loan Duration (months)",
        4,
        72,
        24
    )

    purpose = st.selectbox(
        "Purpose",
        [
            "car",
            "business",
            "education",
            "radio/TV",
            "repairs",
            "vacation/others",
            "furniture/equipment",
            "domestic appliances"
        ]
    )

#COLLECT USER INPUT


input_data = {

    "Age": age,
    "Sex": sex,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving,
    "Checking account": checking,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": purpose

}

#PRDICT BUTTON

predict = st.button("Predict Credit Risk")

#PREDICTION OUTPUT

if predict:

    prediction, probability = predict_risk(input_data)

    with right:

        st.subheader("Prediction Result")

        bad_probability = probability[0] * 100
        good_probability = probability[1] * 100

        if prediction == 1:

            st.success("✅ Low Credit Risk")

            st.metric(
                "Approval Probability",
                f"{good_probability:.2f}%"
            )

        else:

            st.error("❌ High Credit Risk")

            st.metric(
                "Risk Probability",
                f"{bad_probability:.2f}%"
            )

    st.markdown("---")

    st.subheader("📋 Prediction Summary")

    summary = pd.DataFrame({
        "Feature": input_data.keys(),
        "Value": input_data.values()
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # Probability bars, Risk Category and Enhanced Due Diligence code here...
    # PROBABILITY BARS

    st.subheader("Prediction Probability")

    st.write("Good Credit")
    st.progress(min(int(good_probability), 100))
    st.write(f"{good_probability:.2f}%")

    st.write("Bad Credit")
    st.progress(min(int(bad_probability), 100))
    st.write(f"{bad_probability:.2f}%")

    # RISK LEVEL CLASSIFICATION

    st.subheader("Risk Category")

    if bad_probability < 30:
        st.success("🟢 Low Risk")

    elif bad_probability < 60:
        st.warning("🟡 Medium Risk")

    else:
        st.error("🔴 High Risk")

    # ENHANCED DUE DILIGENCE RECOMMENDATION

    if bad_probability >= 60:

        st.markdown("---")

        st.subheader("Enhanced Due Diligence")

        st.warning(
            """
• Verify income documents

• Review previous repayment history

• Request additional financial proof

• Manual credit officer approval recommended

• Consider reducing loan amount
"""
        )

#MODEL COMPARISON

st.markdown("---")

st.subheader("📊 Model Performance Comparison")

comparison = pd.read_csv("evaluation/model_comparison.csv")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

st.subheader("📈 XGBoost Feature Importance")

st.image(
    "images/xgboost_feature_importance.png",
    use_container_width=True
)


#FOOTER

st.markdown("---")

st.markdown(
"""
### About

This project demonstrates an end-to-end machine learning pipeline for
credit risk assessment using the German Credit Dataset.

**Model Used:** XGBoost

**Framework:** Streamlit

**Author:** Arshiya Arif Sayyed
"""
)



