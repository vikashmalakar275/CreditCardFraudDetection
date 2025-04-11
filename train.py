import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import os

def load_data():
    data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    return data

def train_and_save_model():
    data = load_data()
    
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
    
    # Save model and preprocessing objects
    os.makedirs('model', exist_ok=True)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'encoders': label_encoders,
            'scaler': scaler
        }, f)
    
    return model, label_encoders, scaler

if __name__ == "__main__":
    train_and_save_model()