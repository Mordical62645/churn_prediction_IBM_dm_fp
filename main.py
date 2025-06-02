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
df = pd.read(file_path)