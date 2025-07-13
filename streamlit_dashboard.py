import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.decomposition import TruncatedSVD

st.set_page_config(page_title='Bank Marketing Dashboard', layout='wide')

# Load data and model
DATA_PATH = 'Corrected_Kathmandu_Income_Dataset.csv'
MODEL_PATH = 'random_forest_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'
MODEL_COLUMNS_PATH = 'model_columns.pkl'

df = pd.read_csv(DATA_PATH).dropna()
with open(MODEL_PATH, 'rb') as f:
    rf_model = pickle.load(f)
with open(PREPROCESSOR_PATH, 'rb') as f:
    preprocessor = pickle.load(f)
with open(MODEL_COLUMNS_PATH, 'rb') as f:
    model_columns = pickle.load(f)

# For clustering and EDA (as before)
features = ['age', 'education', 'annual_income', 'job']
df_subset = df[features].copy()
try:
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    categorical = [col for col in features if df[col].dtype == 'object']
    numerical = [col for col in features if col not in categorical]
    X = preprocessor.transform(df_subset)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    df['Cluster'] = kmeans_labels
except Exception as e:
    df['Cluster'] = 0

# SVD for 2D cluster visualization
svd = TruncatedSVD(n_components=2)
X_2d = svd.fit_transform(preprocessor.transform(df_subset))
df['SVD1'] = X_2d[:, 0]
df['SVD2'] = X_2d[:, 1]

# Feature importance (from model)
try:
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)
except Exception as e:
    feat_imp_df = pd.DataFrame({'Feature': [], 'Importance': []})

# Sidebar navigation
st.sidebar.title('Navigation')
section = st.sidebar.radio('Go to', ['Overview', 'EDA', 'Clustering', 'Feature Importance', 'Prediction'])

if section == 'Overview':
    st.title('Bank Marketing Data Dashboard')
    st.header('Key Statistics')
    col1, col2, col3 = st.columns(3)
    col1.metric('Total Customers', len(df))
    col2.metric('Subscription Rate', f"{100 * (df['y'] == 'yes').mean():.2f}%")
    col3.metric('Number of Clusters', df['Cluster'].nunique())
    st.subheader('Subscription by Job')
    fig1 = px.bar(df, x='job', color='y', barmode='group', title='Subscription by Job')
    st.plotly_chart(fig1, use_container_width=True)
    st.subheader('Subscription by Education')
    fig2 = px.bar(df, x='education', color='y', barmode='group', title='Subscription by Education')
    st.plotly_chart(fig2, use_container_width=True)

elif section == 'EDA':
    st.title('Exploratory Data Analysis')
    st.subheader('Distributions')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.plotly_chart(px.histogram(df, x='age', nbins=30, title='Age Distribution'), use_container_width=True)
    with col2:
        st.plotly_chart(px.histogram(df, x='balance', nbins=30, title='Balance Distribution'), use_container_width=True)
    with col3:
        st.plotly_chart(px.histogram(df, x='annual_income', nbins=30, title='Annual Income Distribution'), use_container_width=True)
    st.subheader('Boxplots by Subscription')
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(px.box(df, x='y', y='age', title='Age by Subscription'), use_container_width=True)
    with col5:
        st.plotly_chart(px.box(df, x='y', y='balance', title='Balance by Subscription'), use_container_width=True)
    with col6:
        st.plotly_chart(px.box(df, x='y', y='annual_income', title='Annual Income by Subscription'), use_container_width=True)

elif section == 'Clustering':
    st.title('Customer Segmentation (Clustering)')
    st.subheader('2D Cluster Visualization')
    fig = px.scatter(df, x='SVD1', y='SVD2', color='Cluster', symbol='Cluster',
                     hover_data=['age', 'annual_income', 'job', 'education', 'y'],
                     title='Customer Segments (SVD)')
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Cluster Summary')
    cluster_summary = df.groupby('Cluster').agg({
        'age': 'mean',
        'annual_income': 'mean',
        'balance': 'mean',
        'y': lambda x: (x == 'yes').mean()
    }).reset_index().rename(columns={'y': 'subscription_rate'}).round(2)
    st.dataframe(cluster_summary)

elif section == 'Feature Importance':
    st.title('Feature Importance (Random Forest)')
    fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                 title='Feature Importances (Random Forest)')
    st.plotly_chart(fig, use_container_width=True)

elif section == 'Prediction':
    st.title('Predict Subscription to Term Deposit')
    with st.form('prediction_form'):
        age = st.number_input('Age', min_value=18, max_value=100, value=30)
        job = st.selectbox('Job', df['job'].unique())
        job_type = st.selectbox('Job Type', df['job_type'].unique())
        marital = st.selectbox('Marital Status', df['marital'].unique())
        education = st.selectbox('Education', df['education'].unique())
        default = st.selectbox('Default', df['default'].unique())
        balance = st.number_input('Balance', value=0)
        housing = st.selectbox('Housing Loan', df['housing'].unique())
        loan = st.selectbox('Personal Loan', df['loan'].unique())
        contact = st.selectbox('Contact', df['contact'].unique())
        day = st.number_input('Day of Month', min_value=1, max_value=31, value=1)
        month = st.selectbox('Month', df['month'].unique())
        duration = st.number_input('Call Duration', value=0)
        campaign = st.number_input('Campaign Contacts', value=1)
        pdays = st.number_input('Days Since Last Contact', value=-1)
        previous = st.number_input('Previous Contacts', value=0)
        poutcome = st.selectbox('Previous Outcome', df['poutcome'].unique())
        annual_income = st.number_input('Annual Income', value=0)
        submitted = st.form_submit_button('Predict')
    if submitted:
        input_dict = {
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
        }
        input_df = pd.DataFrame([input_dict])
        # Ensure all columns are present and in the right order
        for col in model_columns:
            if col not in input_df.columns:
                if col in df.select_dtypes(include='object').columns:
                    input_df[col] = df[col].mode()[0]
                else:
                    input_df[col] = 0
        input_df = input_df[model_columns]
        input_X = preprocessor.transform(input_df)
        pred = rf_model.predict(input_X)[0]
        proba = rf_model.predict_proba(input_X)[0][1]
        if pred == 1:
            st.success(f'Prediction: YES (Probability: {proba:.2%})')
        else:
            st.error(f'Prediction: NO (Probability: {1-proba:.2%})') 