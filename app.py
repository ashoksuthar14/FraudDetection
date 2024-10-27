import streamlit as st
import pandas as pd
from joblib import load

# Load the trained IsolationForest model (including preprocessor)
@st.cache_resource
def load_model():
    return load('isolation_forest_model.pkl')

model = load_model()

# Streamlit app setup
st.title("Car Insurance Fraud Detection")
st.write("Enter details to check for potential fraud in car insurance claims.")

# Input fields for user
claim_amount = st.number_input("Claim Amount", min_value=0.0, step=0.01)
repair_costs = st.number_input("Repair Costs", min_value=0.0, step=0.01)
type_of_accident = st.selectbox("Type of Accident", ["Theft", "Collision", "Vandalism"])
claim_frequency = st.number_input("Claim Frequency per Policyholder", min_value=0)
time_since_policy_issue = st.number_input("Time Since Policy Issue (months)", min_value=0.0, step=0.1)

# Predict fraud button
if st.button("Predict Fraud"):
    # Collect input into a DataFrame
    input_data = pd.DataFrame({
        'claim_amount': [claim_amount],
        'repair_costs': [repair_costs],
        'type_of_accident': [type_of_accident],
        'claim_frequency_per_policyholder': [claim_frequency],
        'time_since_policy_issue': [time_since_policy_issue]
    })

    # Use the loaded model to make a prediction
    prediction = model.predict(input_data)
    is_fraud = prediction[0] == -1  # -1 in IsolationForest means anomaly/fraud

    # Display results
    if is_fraud:
        st.write("⚠️ **Potential Fraud Detected!**")
    else:
        st.write("✅ **No Fraud Detected**")
