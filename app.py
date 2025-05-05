import sys
sys.path.append(r'C:/Users/rahul barun/OneDrive/Desktop/PROJECT')

import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess import preprocess_data
from optimization import optimize_model
from evaluator import evaluate_model
from models import get_model
from visualizer import plot_metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score
from visualizer import plot_feature_importance


# Function to calculate Precision, Recall, and F1-score
def calculate_additional_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return precision, recall, f1

# Function to perform cross-validation
def cross_validate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
    return cv_scores.mean(), cv_scores.std()  # Mean and std deviation of accuracy

# Function to plot confusion matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st

def plot_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title(f"Confusion Matrix for {model_name}")
    st.pyplot(fig)  #  Replaces plt.show()



# Streamlit App Interface
st.title("Model Zoo with Optuna")

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess the data
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# List of models
model_names = ["logistic_regression", "random_forest", "xgboost", "adaboost", "svm"]

# Results list
results = []

# Loop through models and optimize
for model_name in model_names:
    st.write(f"🔍 Optimizing {model_name}...")
    trial = optimize_model(model_name, X_train, X_test, y_train, y_test)
    model = get_model(model_name)(**trial.params) if trial else get_model(model_name)()
    model.fit(X_train, y_train)
    
    # Standard Evaluation
    accuracy = model.score(X_test, y_test)
    
    # Additional Metrics
    precision, recall, f1 = calculate_additional_metrics(model, X_test, y_test)
    
    # Cross-Validation
    mean_cv, std_cv = cross_validate_model(model, X, y)
    
    # Store results
    results.append({
        "model": model_name, 
        "accuracy": accuracy, 
        "precision": precision, 
        "recall": recall, 
        "f1_score": f1, 
        "cv_mean": mean_cv, 
        "cv_std": std_cv
    })
    
    # Plot Confusion Matrix
    plot_confusion_matrix(model, X_test, y_test, model_name)
    plot_feature_importance(model, data.feature_names, model_name)


# Display Results as DataFrame
df = pd.DataFrame(results)
st.write(df)

# Visualize the results
st.write("### Model Performance Comparison")
plot_metrics(df)
