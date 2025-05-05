
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np

def plot_metrics(results_df):
    """
    Plots a bar chart comparing the accuracy of different models.

    Args:
        results_df (pd.DataFrame): DataFrame containing model names and their respective accuracy scores.

    Returns:
        None: Displays a bar chart using Streamlit.
    """
    # Set the figure size for the plot
    plt.figure(figsize=(10, 6))

    # Create a barplot with model names on the x-axis and accuracy on the y-axis
    sns.barplot(data=results_df, x='model', y='accuracy')

    # Set the title of the plot
    plt.title('Model Accuracy Comparison')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Adjust the layout to avoid clipping of labels
    plt.tight_layout()

    # Show the plot (with Streamlit, we don't need plt.show(), we use st.pyplot)
    st.pyplot(plt)


def plot_feature_importance(model, feature_names, model_name):
    """
    Plots the feature importance of a trained model (if available).

    Args:
        model (sklearn.base.BaseEstimator): The trained machine learning model.
        feature_names (list): List of feature names.
        model_name (str): The name of the model (for labeling the plot).

    Returns:
        None: Displays the feature importance plot or a warning message if not available.
    """
    # Check if the model has the attribute "feature_importances_"
    if hasattr(model, "feature_importances_"):
        # Get the feature importances from the model
        importances = model.feature_importances_

        # Sort the feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Create a plot to display the feature importances
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Bar chart of feature importances
        ax.bar(range(len(importances)), importances[indices])

        # Set the labels of the x-axis to the sorted feature names
        ax.set_xticks(range(len(importances)))
        ax.set_xticklabels(np.array(feature_names)[indices], rotation=45, ha='right')

        # Set the title of the plot
        ax.set_title(f"Feature Importance - {model_name}")

        # Adjust layout to avoid label clipping
        plt.tight_layout()

        # Display the plot in the Streamlit app
        st.pyplot(fig)
    else:
        # If the model does not support feature importance, display a warning
        st.warning(f" {model_name} does not support feature importance.")
