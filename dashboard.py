import streamlit as st
import pandas as pd
import pickle

st.title("Bank Customer Subscription Prediction Dashboard")

# Load preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature columns and placeholder values (from first row of CSV)
default_values = {
    'age': 24,
    'job': 'bank_employee',
    'job_type': 'White-collar',
    'marital': 'married',
    'education': 'slc',
    'default': 'no',
    'balance': 66661,
    'housing': 'no',
    'loan': 'yes',
    'contact': 'Ncell',
    'day': 3,
    'month': 'jun',
    'duration': 1115,
    'campaign': 5,
    'pdays': 15,
    'previous': 3,
    'poutcome': 'not_contacted',
    'annual_income': 168480
}

# Options for categorical fields (from your dataset)
job_options = [
    'bank_employee', 'government_job', 'unemployed', 'IT_professional', 'teacher',
    'shopkeeper', 'private_office', 'construction_worker', 'driver', 'farmer'
]
job_type_options = ['White-collar', 'Blue-collar', 'Business', 'Unemployed']
marital_options = ['married', 'single', 'divorced']
education_options = ['none', 'slc', '+2', 'bachelor', 'masters', 'PhD']
default_options = ['no', 'yes']
housing_options = ['no', 'yes']
loan_options = ['no', 'yes']
contact_options = ['Ncell', 'NTC', 'field_visit', 'unknown']
month_options = [
    'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
]
poutcome_options = ['not_contacted', 'unsuccessful', 'successful']

# User input widgets
st.header("Enter Customer Information:")
age = st.number_input("Age", min_value=18, max_value=100, value=default_values['age'])
job = st.selectbox("Job", job_options, index=job_options.index(default_values['job']))
job_type = st.selectbox("Job Type", job_type_options, index=job_type_options.index(default_values['job_type']))
marital = st.selectbox("Marital Status", marital_options, index=marital_options.index(default_values['marital']))
education = st.selectbox("Education", education_options, index=education_options.index(default_values['education']))
default = st.selectbox("Defaulted Before?", default_options, index=default_options.index(default_values['default']))
balance = st.number_input("Balance", value=default_values['balance'])
housing = st.selectbox("Housing Loan?", housing_options, index=housing_options.index(default_values['housing']))
loan = st.selectbox("Personal Loan?", loan_options, index=loan_options.index(default_values['loan']))
contact = st.selectbox("Contact Method", contact_options, index=contact_options.index(default_values['contact']))
day = st.number_input("Day of Month Contacted", min_value=1, max_value=31, value=default_values['day'])
month = st.selectbox("Month Contacted", month_options, index=month_options.index(default_values['month']))
duration = st.number_input("Last Contact Duration (seconds)", value=default_values['duration'])
campaign = st.number_input("Number of Contacts During Campaign", value=default_values['campaign'])
pdays = st.number_input("Days Since Last Contact (-1 if never)", value=default_values['pdays'])
previous = st.number_input("Number of Previous Contacts", value=default_values['previous'])
poutcome = st.selectbox("Previous Campaign Outcome", poutcome_options, index=poutcome_options.index(default_values['poutcome']))
annual_income = st.number_input("Annual Income (NPR)", value=default_values['annual_income'])

# Predict button
if st.button("Predict Subscription"):
    # Create DataFrame for prediction
    new_customer = pd.DataFrame([{
        'age': age,
        'job': job,
        'job_type': job_type,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'annual_income': annual_income
    }])
    # Preprocess and predict
    X_new = preprocessor.transform(new_customer)
    prediction = model.predict(X_new)
    prediction_proba = model.predict_proba(X_new)
    # Output
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"This customer is likely to JOIN the bank. Probability: {prediction_proba[0][1]:.2f}")
    else:
        st.warning(f"This customer is NOT likely to join. Probability: {prediction_proba[0][1]:.2f}") 