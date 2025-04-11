import streamlit as st
import pandas as pd
import pickle
from visualize import (
    plot_churn_distribution,
    plot_numerical_distribution,
    plot_categorical_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_confusion_matrix,
    load_model
)
from train import load_data

# Set page config
st.set_page_config(
    page_title="Customer Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load data and model
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model():
    return load_model()

# Sidebar navigation
st.sidebar.title("App Sections")
st.sidebar.image("images/dashboard.png", width=200)
page = st.sidebar.radio("Go to", ["About App", "Churn Prediction", "Data Analysis & Model Performance"], index=0)

# About App Page
if page == "About App":
    st.title("Credit Card Fraud Detection App")
    st.image("images/card.png", use_container_width=True)
    
    st.markdown("""
    ## Purpose of the App
    The **Customer Analytics Dashboard** is designed to help businesses understand their customers better and take proactive measures to improve customer retention. This app leverages machine learning to predict customer churn and provides insights into key factors influencing customer behavior.

    ## Key Features
    - **Data Analysis**: Visualize customer data patterns, including churn distribution, numerical feature trends, and categorical feature distributions.
    - **Model Performance**: Evaluate the machine learning model's performance using metrics like feature importance and confusion matrix.
    - **Churn Prediction**: Predict whether a customer is likely to churn based on their details.

    ## How to Use
    1. **Navigate**: Use the sidebar to switch between different sections:
        - **About App**: Learn about the app's purpose and features.
        - **Churn Prediction**: Predict churn for a new customer by filling out a form.
        - **Data Analysis & Model Performance**: Explore the dataset and evaluate the model's performance.
    2. **Analyze Data**: Go to the "Data Analysis & Model Performance" section to explore customer data and understand key trends.
    3. **Predict Churn**: Use the "Churn Prediction" section to input customer details and get a prediction on whether they are likely to churn.

    ## Benefits
    - **Proactive Retention**: Identify customers at risk of churning and take timely actions to retain them.
    - **Data-Driven Decisions**: Make informed decisions based on customer data insights.
    - **Improved Customer Experience**: Understand customer needs and preferences to enhance their experience.

    ## Contact
    For any queries or feedback, feel free to reach out:
    - 📧 **Email 1**: [g24ait2204@iitj.ac.in](mailto:g24ait2204@iitj.ac.in)
    - 📧 **Email 2**: [g24ait2008@iitj.ac.in](mailto:g24ait2008@iitj.ac.in)
    """)

# Data Analysis Page
elif page == "Data Analysis & Model Performance":
    st.title("Data Analysis")
    data = get_data()
    
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {data.shape}")
    
    # Visualizations
    st.subheader("Data Visualizations")
    
    # Churn distribution
    st.write("### Churn Distribution")
    st.pyplot(plot_churn_distribution(data))
    
    # Numerical features distribution
    st.write("### Numerical Features Distribution")
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    selected_num = st.selectbox("Select numerical feature", num_cols)
    st.pyplot(plot_numerical_distribution(data, selected_num))
    
    # Categorical features distribution
    st.write("### Categorical Features Distribution")
    cat_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'customerID']
    selected_cat = st.selectbox("Select categorical feature", cat_cols)
    st.pyplot(plot_categorical_distribution(data, selected_cat))
    
    # Correlation heatmap
    st.write("### Correlation Heatmap")
    st.pyplot(plot_correlation_heatmap(data))

    st.title("Customer Churn Prediction")
    data = get_data()
    model_data = get_model()
    
    # Model evaluation
    st.subheader("Model Performance")
    
    st.write("### Feature Importance")
    st.pyplot(plot_feature_importance(model_data))
    
    st.write("### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(model_data, data))

# Churn Prediction Page
elif page == "Churn Prediction":
    data = get_data()
    model_data = get_model()  
    # Prediction form
    st.subheader("Predict Churn for New Customer")
    with st.form("prediction_form"):
        st.write("### Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            senior_citizen = 1 if senior_citizen == "Yes" else 0
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
            for col in [k for k in input_data.keys() if k in model_data['encoders']]:
                input_df[col] = model_data['encoders'][col].transform(input_df[col])
            
            # Scale numerical features
            numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
            input_df[numerical_cols] = model_data['scaler'].transform(input_df[numerical_cols])
            
            # Make prediction
            prediction = model_data['model'].predict(input_df)
            probability = model_data['model'].predict_proba(input_df)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error(f"Prediction: Churn (Probability: {probability[0][1]:.2%})")
                st.write("This customer is likely to churn. Consider retention strategies.")
            else:
                st.success(f"Prediction: No Churn (Probability: {probability[0][0]:.2%})")
                st.write("This customer is likely to stay.")


# Footer Section
st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f9f9f9;
            text-align: center;
            height: 40px; /* Adjusted height for a thinner footer */
            line-height: 40px; /* Align text vertically */
            border-top: 1px solid #eaeaea;
            font-size: 14px; /* Smaller font size */
        }
        .footer a {
            color: #0073b1; /* LinkedIn blue color */
            text-decoration: none;
            margin-left: 5px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">
    <div class="footer">
        <p>
            Connect with us on 
            <a href="https://www.linkedin.com/in/vikash-malakar-25aa90116/" target="_blank">
                <i class="fab fa-linkedin"></i> Vikash Malakar
            </a> 
            and 
            <a href="https://www.linkedin.com/in/netajik/" target="_blank">
                <i class="fab fa-linkedin"></i> Netaji K
            </a> 
            | © 2025 Credit Card Fraud Detection App
        </p>
    </div>
    """,
    unsafe_allow_html=True
)