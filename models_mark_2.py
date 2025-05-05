# models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier

def get_model(model_name):
    if model_name == "logistic_regression":
        return LogisticRegression
    elif model_name == "random_forest":
        return RandomForestClassifier
    elif model_name == "xgboost":
        return XGBClassifier
    elif model_name == "svm":
        return SVC
    elif model_name == "adaboost":
        return AdaBoostClassifier
    else:
        return None  # If the model name doesn't match any option
