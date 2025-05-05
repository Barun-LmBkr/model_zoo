import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(X, y, scale=True):
    if isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return train_test_split(X, y, test_size=0.2, random_state=42)
