# Model Zoo with Optuna

This project is a machine learning model comparison tool using **Optuna** for hyperparameter optimization. It allows users to upload their own datasets, select target columns, and test various machine learning models. The models are optimized using Optuna, evaluated using multiple metrics, and visualized with various plots. The best model is selected and can be downloaded for further use.

**Branch**: `main` (or mention any other active branch you are working on).

## Features

- **Dataset Upload**: Users can upload their own CSV dataset for analysis.
- **Model Selection**: Built-in support for several common ML algorithms such as:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - AdaBoost
  - SVM
- **Hyperparameter Optimization**: Uses Optuna to automatically tune hyperparameters for each model.
- **Model Evaluation**: Models are evaluated on:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Cross-validation
- **Visualization**: Confusion matrix and feature importance are plotted for each model.
- **Download Best Model**: The best model can be saved as a `.pkl` file for later use.

## Installation

Installation
To get started with this project, you need to clone it from GitHub and install the necessary dependencies.

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/model_zoo.git
   cd model_zoo

2. Model Selection and Hyperparameter Optimization
The tool will automatically select models and optimize their hyperparameters using Optuna.

Supported models include:

Logistic Regression

Random Forest

XGBoost

AdaBoost

SVM

3. Model Evaluation
After training, the app will display the following metrics:

Accuracy

Precision

Recall

F1-Score

Cross-Validation (mean and standard deviation)

4. Model Visualization
You will be able to visualize:

The Confusion Matrix of each model.

Feature Importance for models that support it (e.g., Random Forest, XGBoost).

5. Model Download
Once the evaluation is complete, the best model is saved as a .pkl file. You can download it for future use.

Limitations
Data format: The tool assumes that the dataset is in CSV format and that the target column is categorical. It might not work properly with datasets that have missing or non-numeric values in feature columns.

Model Compatibility: While most common models are supported, some complex models or custom implementations may not work with this framework.

Performance on Large Datasets: The optimization process can be slow on large datasets due to the exhaustive search for hyperparameters. Consider using a smaller subset of the data for faster results.

No Support for Time Series: This tool is designed for classification tasks and may not perform well with time series data or sequential models.

License
This project is licensed under the MIT License - see the LICENSE file for details
