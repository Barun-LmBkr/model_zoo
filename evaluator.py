
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of a model using different metrics: accuracy, F1-score, and ROC AUC.

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        X_test (pd.DataFrame or np.ndarray): Feature data for testing.
        y_test (pd.Series or np.ndarray): Target labels for testing.

    Returns:
        dict: A dictionary containing the model's performance metrics:
            - "accuracy": Accuracy score
            - "f1_score": Weighted F1-score
            - "roc_auc": ROC AUC score (if applicable)
    """
    
    # Get model predictions for the test data
    preds = model.predict(X_test)
    
    # Create a dictionary to hold the evaluation metrics
    evaluation_metrics = {
        # Accuracy score of the model
        "accuracy": accuracy_score(y_test, preds),
        
        # Weighted F1-score
        "f1_score": f1_score(y_test, preds, average='weighted'),
        
        # ROC AUC score, but only if the model has predict_proba() method
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) 
                     if hasattr(model, "predict_proba") else None
    }
    
    # Return the dictionary containing the evaluation metrics
    return evaluation_metrics
