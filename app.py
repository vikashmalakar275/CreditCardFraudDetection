import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import pickle
import os

# Set page config
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data function with caching
@st.cache_data
def load_data():
    data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Data cleaning
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    return data

# Train model function with caching
@st.cache_resource
def train_model(data):
    # Preprocessing
    data = data.copy()
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'customerID']
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    # Split data
    X = data.drop(['Churn', 'customerID'], axis=1)
    y = data['Churn']
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_res, y_res)
    
    return model, label_encoders, scaler

# Save model function
def save_model(model, encoders, scaler):
    with open('model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'encoders': encoders,
            'scaler': scaler
        }, f)

# Load model function
def load_model():
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            return pickle.load(f)
    return None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Churn Prediction"])

# Home Page
if page == "Home":
    st.title("Customer Analytics Dashboard")
    st.image("https://cdn-icons-png.flaticon.com/512/2093/2093638.png", width=200)
    
    st.markdown("""
    ## Welcome to the Customer Analytics Dashboard
    
    This application helps you:
    - Analyze customer data patterns
    - Predict customer churn
    - Understand key factors affecting customer retention
    
    Use the navigation panel on the left to explore different sections.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Analysis")
        st.markdown("""
        - Explore the dataset
        - View visualizations
        - Understand data distributions
        """)
    
    with col2:
        st.subheader("Churn Prediction")
        st.markdown("""
        - Predict customer churn
        - View model performance
        - Understand feature importance
        """)

# Data Analysis Page
elif page == "Data Analysis":
    st.title("Data Analysis")
    data = load_data()
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {data.shape}")
    st.dataframe(data.head())
    
    st.subheader("Data Description")
    st.write(data.describe())
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    # Churn distribution
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='Churn', data=data, ax=ax)
    st.pyplot(fig)
    
    # Numerical features distribution
    st.write("### Numerical Features Distribution")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_num = st.selectbox("Select numerical feature", num_cols)
    fig, ax = plt.subplots()
    sns.histplot(data[selected_num], kde=True, ax=ax)
    st.pyplot(fig)
    
    # Categorical features distribution
    st.write("### Categorical Features Distribution")
    cat_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'customerID']
    selected_cat = st.selectbox("Select categorical feature", cat_cols)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.countplot(x=selected_cat, hue='Churn', data=data, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_data = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    # Model training section
    st.subheader("Model Training Information")
    st.markdown("""
    ### How We Train the Model
    
    1. **Data Preprocessing**:
       - Handle missing values
       - Encode categorical variables
       - Scale numerical features
    
    2. **Handling Class Imbalance**:
       - Use SMOTE (Synthetic Minority Over-sampling Technique)
    
    3. **Model Selection**:
       - Random Forest Classifier
       - 100 estimators
       - Default hyperparameters
    
    4. **Evaluation Metrics**:
       - Accuracy
       - Precision
       - Recall
       - F1-score
    """)

# Churn Prediction Page
elif page == "Churn Prediction":
    st.title("Customer Churn Prediction")
    data = load_data()
    
    # Load or train model
    model_data = load_model()
    if model_data:
        model = model_data['model']
        label_encoders = model_data['encoders']
        scaler = model_data['scaler']
        st.success("Using pre-trained model")
    else:
        with st.spinner("Training model... This may take a few minutes"):
            model, label_encoders, scaler = train_model(data)
            save_model(model, label_encoders, scaler)
        st.success("Model trained successfully!")
    
    # Model evaluation
    st.subheader("Model Performance")
    
    # Prepare data for evaluation
    data_copy = data.copy()
    data_copy['Churn'] = data_copy['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = [col for col in data_copy.columns if data_copy[col].dtype == 'object' and col != 'customerID']
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Encode and scale
    for col in categorical_cols:
        data_copy[col] = label_encoders[col].transform(data_copy[col])
    
    data_copy[numerical_cols] = scaler.transform(data_copy[numerical_cols])
    
    X = data_copy.drop(['Churn', 'customerID'], axis=1)
    y = data_copy['Churn']
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    st.write("### Classification Report")
    st.text(classification_report(y, y_pred))
    
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Feature importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
    st.pyplot(fig)
    
    # Prediction form
    st.subheader("Predict Churn for New Customer")
    with st.form("prediction_form"):
        st.write("### Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        
        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        
        col3, col4 = st.columns(2)
        
        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        
        with col4:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0, max_value=200, value=50)
            total_charges = st.number_input("Total Charges ($)", min_value=0, max_value=10000, value=1000)
        
        submitted = st.form_submit_button("Predict Churn")
        
        if submitted:
            # Prepare input data
            input_data = {
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col in categorical_cols:
                input_df[col] = label_encoders[col].transform(input_df[col])
            
            # Scale numerical features
            input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(input_df.drop(['customerID'], axis=1, errors='ignore'))
            probability = model.predict_proba(input_df.drop(['customerID'], axis=1, errors='ignore'))
            
            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"Prediction: Churn (Probability: {probability[0][1]:.2%})")
                st.write("This customer is likely to churn. Consider retention strategies.")
            else:
                st.success(f"Prediction: No Churn (Probability: {probability[0][0]:.2%})")
                st.write("This customer is likely to stay.")