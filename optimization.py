import optuna
from sklearn.metrics import accuracy_score
from models import get_model

def optimize_model(model_name, X_train, X_test, y_train, y_test):
    """
    Optimizes hyperparameters of a given model using Optuna.

    Args:
        model_name (str): The name of the model to optimize (e.g., 'random_forest', 'xgboost', 'svm').
        X_train (pd.DataFrame or np.ndarray): Feature data for training.
        X_test (pd.DataFrame or np.ndarray): Feature data for testing.
        y_train (pd.Series or np.ndarray): Target labels for training.
        y_test (pd.Series or np.ndarray): Target labels for testing.

    Returns:
        optuna.study.Study: The optimized trial study with the best hyperparameters.
    """
    
    # Define the objective function for Optuna optimization
    def objective(trial):
        """
        The objective function that Optuna tries to minimize or maximize.
        It suggests hyperparameters and evaluates the model's performance.

        Args:
            trial (optuna.trial.Trial): The trial object from Optuna that suggests hyperparameters.

        Returns:
            float: The model's accuracy score on the test data.
        """
        # Hyperparameter tuning for 'random_forest'
        if model_name == "random_forest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200)  # Number of trees
            max_depth = trial.suggest_int("max_depth", 2, 32)  # Maximum depth of each tree
            model = get_model(model_name)(n_estimators=n_estimators, max_depth=max_depth)
        
        # Hyperparameter tuning for 'xgboost'
        elif model_name == "xgboost":
            eta = trial.suggest_float("eta", 0.01, 0.3)  # Learning rate
            model = get_model(model_name)(eta=eta)
        
        # Hyperparameter tuning for 'svm'
        elif model_name == "svm":
            c = trial.suggest_float("C", 0.1, 10.0)  # Regularization parameter
            model = get_model(model_name)(C=c)
        
        # For any other model, use default settings
        else:
            model = get_model(model_name)()
        
        # Fit the model to the training data
        model.fit(X_train, y_train)
        
        # Predict using the test data
        preds = model.predict(X_test)
        
        # Calculate and return the accuracy score of the model
        return accuracy_score(y_test, preds)

    # Create an Optuna study to optimize the objective function
    study = optuna.create_study(direction="maximize")  # We want to maximize the accuracy score
    study.optimize(objective, n_trials=30)  # Perform 30 trials of optimization

    # Return the best trial with the optimized hyperparameters
    return study.best_trial
