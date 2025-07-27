import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

st.set_page_config(layout="wide")
st.title("Bank Customer Analysis Dashboard (Predictive & Prescriptive)")

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('Corrected_Kathmandu_Income_Dataset.csv').dropna()

def predictive_section():
    st.header("Predict Customer Subscription")
    with st.expander("About this tool", expanded=False):
        st.markdown("""
        Enter customer details to predict the likelihood of subscription. The model uses a Random Forest trained on Kathmandu bank marketing data.
        """)
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
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=default_values['age'])
            job = st.selectbox("Job", job_options, index=job_options.index(default_values['job']))
            job_type = st.selectbox("Job Type", job_type_options, index=job_type_options.index(default_values['job_type']))
            marital = st.selectbox("Marital Status", marital_options, index=marital_options.index(default_values['marital']))
            education = st.selectbox("Education", education_options, index=education_options.index(default_values['education']))
        with col2:
            default = st.selectbox("Defaulted Before?", default_options, index=default_options.index(default_values['default']))
            balance = st.number_input("Balance", value=default_values['balance'])
            housing = st.selectbox("Housing Loan?", housing_options, index=housing_options.index(default_values['housing']))
            loan = st.selectbox("Personal Loan?", loan_options, index=loan_options.index(default_values['loan']))
            contact = st.selectbox("Contact Method", contact_options, index=contact_options.index(default_values['contact']))
        with col3:
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

def prescriptive_section():
    st.header("Prescriptive Analysis: Clusters, Feature Importances, and EDA")
    tab1, tab2, tab3, tab4 = st.tabs(["Clusters", "Feature Importances", "EDA", "Raw Data"])

    with tab1:
        st.subheader("Customer Segmentation")
        features = ['age', 'education', 'annual_income', 'job']
        df_subset = df[features].copy()
        categorical = [col for col in features if df[col].dtype == 'object']
        numerical = [col for col in features if col not in categorical]
        preprocessor_cluster = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), categorical),
            ('num', StandardScaler(), numerical)
        ])
        X = preprocessor_cluster.fit_transform(df_subset)
        kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
        labels = kmeans.fit_predict(X)
        df['Cluster'] = labels
        cluster_summary = df.groupby('Cluster').agg({
            'age': ['mean', 'median', 'std'],
            'annual_income': ['mean', 'median', 'std'],
            'balance': ['mean', 'median', 'std'],
            'job': lambda x: x.value_counts().index[0],
            'education': lambda x: x.value_counts().index[0],
            'marital': lambda x: x.value_counts().index[0],
        })
        cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
        st.markdown("**Cluster Profiles:**")
        st.dataframe(cluster_summary)
        cluster_id = st.selectbox("Select Cluster to Explore", sorted(df['Cluster'].unique()))
        cluster_df = df[df['Cluster'] == cluster_id]
        st.markdown(f"**Cluster {cluster_id}**: {len(cluster_df)} customers")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Average Age", f"{cluster_df['age'].mean():.1f}")
            st.metric("Average Annual Income", f"NPR {cluster_df['annual_income'].mean():,.0f}")
            st.metric("Most Common Job", cluster_df['job'].mode()[0])
        with colB:
            st.metric("Average Balance", f"NPR {cluster_df['balance'].mean():,.0f}")
            st.metric("Most Common Education", cluster_df['education'].mode()[0])
            st.metric("Most Common Marital Status", cluster_df['marital'].mode()[0])
        radar_features = ['age', 'annual_income', 'balance']
        means = [cluster_df[f].mean() for f in radar_features]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=means + [means[0]],
                                            theta=radar_features + [radar_features[0]],
                                            fill='toself', name=f'Cluster {cluster_id}'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                showlegend=False, title="Cluster Feature Profile (Radar Chart)")
        st.plotly_chart(fig_radar, use_container_width=True)
        st.markdown("**Sample Customers in this Cluster:**")
        st.dataframe(cluster_df.head(10))
        svd = TruncatedSVD(n_components=2)
        X_2d = svd.fit_transform(X)
        fig2d = px.scatter(x=X_2d[:,0], y=X_2d[:,1], color=labels.astype(str),
                           labels={'x':'Component 1', 'y':'Component 2', 'color':'Cluster'},
                           title="Customer Segments (2D, Interactive)")
        st.plotly_chart(fig2d, use_container_width=True)
        st.info("Explore clusters to identify actionable customer segments for targeted marketing.")

    with tab2:
        st.subheader("Feature Importances (Random Forest Model)")
        importances = model.feature_importances_
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        N = st.slider("Show Top N Features", min_value=3, max_value=len(importances), value=10)
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).head(N)
        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         title="Top Feature Importances", hover_data=['Importance'])
        st.plotly_chart(fig_imp, use_container_width=True)
        st.dataframe(imp_df)
        st.info("Focus on the most important features to improve model performance and business outcomes.")

    with tab3:
        st.subheader("Exploratory Data Analysis (EDA)")
        st.markdown("Interactively explore the data. Use filters to focus on specific segments.")
        with st.expander("Filter Data", expanded=False):
            colf1, colf2, colf3 = st.columns(3)
            with colf1:
                job_filter = st.multiselect("Job", options=sorted(df['job'].unique()), default=sorted(df['job'].unique()))
            with colf2:
                education_filter = st.multiselect("Education", options=sorted(df['education'].unique()), default=sorted(df['education'].unique()))
            with colf3:
                y_filter = st.multiselect("Subscribed?", options=sorted(df['y'].unique()), default=sorted(df['y'].unique()))
        filtered = df[df['job'].isin(job_filter) & df['education'].isin(education_filter) & df['y'].isin(y_filter)]
        colp1, colp2 = st.columns(2)
        with colp1:
            fig_income = px.histogram(filtered, x='annual_income', color='y', barmode='overlay', nbins=30,
                                     title='Annual Income Distribution by Subscription')
            st.plotly_chart(fig_income, use_container_width=True)
            fig_box = px.box(filtered, x='y', y='annual_income', color='y',
                             title='Annual Income by Target Variable')
            st.plotly_chart(fig_box, use_container_width=True)
        with colp2:
            fig_job = px.histogram(filtered, x='job', color='y', barmode='group',
                                  title='Job Distribution by Target Variable')
            st.plotly_chart(fig_job, use_container_width=True)
            fig_edu = px.histogram(filtered, x='education', color='y', barmode='group',
                                   title='Education Distribution by Target Variable')
            st.plotly_chart(fig_edu, use_container_width=True)
        st.markdown("**Summary Statistics (Filtered Data):**")
        st.dataframe(filtered.describe(include='all'))

    with tab4:
        st.dataframe(df)

tabs = st.tabs(["Predictive Analysis", "Prescriptive Analysis"])
with tabs[0]:
    predictive_section()
with tabs[1]:
    prescriptive_section() 