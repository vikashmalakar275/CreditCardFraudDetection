import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from train import load_data
from sklearn.metrics import confusion_matrix

def load_model():
    with open('model/model.pkl', 'rb') as f:
        return pickle.load(f)

def plot_churn_distribution(data):
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=data)
    plt.title('Churn Distribution')
    return plt

def plot_numerical_distribution(data, column):
    plt.figure(figsize=(6, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    return plt

def plot_categorical_distribution(data, column):
    plt.figure(figsize=(10, 4))
    sns.countplot(x=column, hue='Churn', data=data)
    plt.title(f'{column} vs Churn')
    plt.xticks(rotation=45)
    return plt

def plot_correlation_heatmap(data):
    numeric_data = data[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    return plt

def plot_feature_importance(model_data):
    feature_importance = pd.DataFrame({
        'Feature': model_data['model'].feature_names_in_,
        'Importance': model_data['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Important Features')
    return plt

def plot_confusion_matrix(model_data, data):
    # Prepare data
    data = data.copy()
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and col != 'customerID']
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Encode and scale
    for col in categorical_cols:
        data[col] = model_data['encoders'][col].transform(data[col])
    
    data[numerical_cols] = model_data['scaler'].transform(data[numerical_cols])
    
    X = data.drop(['Churn', 'customerID'], axis=1)
    y = data['Churn']
    
    # Predictions
    y_pred = model_data['model'].predict(X)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    return plt

if __name__ == "__main__":
    data = load_data()
    model_data = load_model()
    
    # Example usage
    plot_churn_distribution(data).show()
    plot_numerical_distribution(data, 'tenure').show()
    plot_categorical_distribution(data, 'Contract').show()
    plot_correlation_heatmap(data).show()
    plot_feature_importance(model_data).show()
    plot_confusion_matrix(model_data, data).show()