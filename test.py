#### LIBRARIES ####
import pandas as pd
import numpy as np

# For splitting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# For modeling and evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# For visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For handling imbalanced datasets
from imblearn.over_sampling import SMOTE

# For high-performance models
import xgboost as xgb

# Optional: for building a simple interactive web app
import streamlit as st


#### DATASET ####
# Load CSV
file_path = "IBM.csv"
df = pd.read_csv(file_path)
# print(df)

#### PREPROCESSING ####
# Clean dataset
columns_to_clean = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
    ]
for col in columns_to_clean:
    df[col] = df[col].str.replace(',', '').astype(float)
    
#### EXPLORATORY DATA ANALYSIS (EDA) ####

#### HANDLE CLASS IMBALANCE ####

#### TRAIN TEST SPLIT ####

#### FEATURE SCALING ####

#### MODELLING ####

#### EVALUATION ####

#### PREDICTION AND OUTPUT ####


