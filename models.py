from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

def get_model(name):
    models = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "adaboost": AdaBoostClassifier,
        "xgboost": XGBClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier
    }
    return models.get(name.lower(), None)
