import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(layout="wide")
st.title("Bank Customer Analysis Dashboard (Predictive & Prescriptive)")

# Load preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
df = pd.read_csv('Corrected_Kathmandu_Income_Dataset.csv').dropna()

# --- Predictive Section (same as dashboard.py) ---
def predictive_section():
    st.header("Predict Customer Subscription")
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

    with st.form("predict_form"):
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
        submit = st.form_submit_button("Predict Subscription")

    if submit:
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
        X_new = preprocessor.transform(new_customer)
        prediction = model.predict(X_new)
        prediction_proba = model.predict_proba(X_new)
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"This customer is likely to JOIN the bank. Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.warning(f"This customer is NOT likely to join. Probability: {prediction_proba[0][1]:.2f}")

# --- Prescriptive Section ---
def prescriptive_section():
    st.header("Prescriptive Analysis: Clusters, Feature Importances, and EDA")
    tab1, tab2, tab3, tab4 = st.tabs(["Clusters", "Feature Importances", "EDA", "Raw Data"])

    # --- Clusters Tab ---
    with tab1:
        st.subheader("Customer Segmentation (KMeans Clustering)")
        features = ['age', 'education', 'annual_income', 'job']
        df_subset = df[features].copy()
        categorical = [col for col in features if df[col].dtype == 'object']
        numerical = [col for col in features if col not in categorical]
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        preprocessor_cluster = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), categorical),
            ('num', StandardScaler(), numerical)
        ])
        X = preprocessor_cluster.fit_transform(df_subset)
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        df['Cluster'] = labels
        # Cluster summary
        cluster_summary = df.groupby('Cluster').agg({
            'age': 'mean',
            'annual_income': 'mean',
            'balance': 'mean'
        }).reset_index()
        st.dataframe(cluster_summary)
        # Cluster counts
        cluster_counts = df['Cluster'].value_counts().sort_index()
        cluster_labels = [f"Cluster {i}" for i in cluster_counts.index]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        sns.barplot(x=cluster_labels, y=cluster_counts.values, ax=ax1, palette="viridis")
        ax1.set_ylabel("Number of Customers", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        st.pyplot(fig)
        # 2D visualization
        svd = TruncatedSVD(n_components=2)
        X_2d = svd.fit_transform(X)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
        legend1 = ax2.legend(*scatter.legend_elements(), title="Cluster")
        ax2.add_artist(legend1)
        ax2.set_title('Customer Segments (2D)')
        ax2.set_xlabel('Component 1')
        ax2.set_ylabel('Component 2')
        st.pyplot(fig2)

    # --- Feature Importances Tab ---
    with tab2:
        st.subheader("Feature Importances (Random Forest Model)")
        importances = model.feature_importances_
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=True)
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.barh(imp_df['Feature'], imp_df['Importance'])
        ax3.set_title("Feature Importances")
        st.pyplot(fig3)
        st.dataframe(imp_df.sort_values(by='Importance', ascending=False))

    # --- EDA Tab ---
    with tab3:
        st.subheader("Exploratory Data Analysis (EDA)")
        # Distribution of annual income
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['annual_income'], kde=True, ax=ax4)
        ax4.set_title('Distribution of Annual Income')
        st.pyplot(fig4)
        # Boxplots and countplots by target variable
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='y', y='annual_income', data=df, ax=ax5)
        ax5.set_title('Annual Income by Target Variable')
        st.pyplot(fig5)
        fig6, ax6 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='job', hue='y', data=df, ax=ax6)
        ax6.set_title('Job Distribution by Target Variable')
        ax6.tick_params(axis='x', rotation=45)
        st.pyplot(fig6)
        fig7, ax7 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='marital', hue='y', data=df, ax=ax7)
        ax7.set_title('Marital Distribution by Target Variable')
        st.pyplot(fig7)
        fig8, ax8 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='education', hue='y', data=df, ax=ax8)
        ax8.set_title('Education Distribution by Target Variable')
        ax8.tick_params(axis='x', rotation=45)
        st.pyplot(fig8)
        fig9, ax9 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='default', hue='y', data=df, ax=ax9)
        ax9.set_title('Default Distribution by Target Variable')
        st.pyplot(fig9)
        fig10, ax10 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='housing', hue='y', data=df, ax=ax10)
        ax10.set_title('Housing Loan by Target Variable')
        st.pyplot(fig10)
        fig11, ax11 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='loan', hue='y', data=df, ax=ax11)
        ax11.set_title('Personal Loan by Target Variable')
        st.pyplot(fig11)
        fig12, ax12 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='month', hue='y', data=df, ax=ax12)
        ax12.set_title('Month by Target Variable')
        ax12.tick_params(axis='x', rotation=45)
        st.pyplot(fig12)
        fig13, ax13 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='y', data=df, ax=ax13)
        ax13.set_title('Target Variable Distribution')
        st.pyplot(fig13)
        fig14, ax14 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='poutcome', hue='y', data=df, ax=ax14)
        ax14.set_title('Poutcome by Target Variable')
        st.pyplot(fig14)

    # --- Raw Data Tab ---
    with tab4:
        st.dataframe(df)

# --- Main Layout ---
tabs = st.tabs(["Predictive Analysis", "Prescriptive Analysis"])
with tabs[0]:
    predictive_section()
with tabs[1]:
    prescriptive_section() 