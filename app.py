import streamlit as st
import pandas as pd
import joblib

# Load trained model
clf = joblib.load("GradientBoostingAdoption_model.pkl")

st.title("⚡ Adoption Likelihood Predictor")

st.write("Fill in the details below and click *Predict* to see the result.")

# User inputs
respondent_type = st.selectbox("Respondent Type", ["Household", "Business", "Community"])
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
num_employees = st.number_input("Number of Employees", min_value=0, max_value=500, value=10)
monthly_income = st.number_input("Monthly Household Income", min_value=0, max_value=1_000_000, value=50000, step = 1000)
energy_source = st.selectbox("Energy Source", ["National Grid", "Solar", "Fossil generator", "Mixed sources", "National Grid + Solar","National Grid + Fossil generator","National Grid + Mixed sources"])
willingness_to_pay = st.number_input("Willingness to Pay (₦)", min_value=0, max_value=1_000_000, value=20000, step=1000)


# When user clicks button
if st.button("🔮 Predict Adoption Likelihood"):
    # Create dataframe for model
    input_df = pd.DataFrame({
        "respondent_type": [respondent_type],
        "household_size": [household_size],
        "num_employees": [num_employees],
        "monthly_household_income": [monthly_income],
        "energy_source": [energy_source],
        "willingness_to_pay": [willingness_to_pay]
    })

    # Encode categorical features same as training
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = input_df[col].astype("category").cat.codes

    # Make prediction
    prediction = clf.predict(input_df)[0]

    # Show result
    if prediction == 1:
        st.success("✅ Likely to Adopt")
    else:
        st.error("❌ Not Likely to Adopt")
