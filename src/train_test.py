import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def adjusted_r2(r2, n, p):
    """Calculate Adjusted R²."""
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))

def prepare_data(df, target, test_threshold):
    
    # Features & Target
    y = df.loc[:,target]
    X = df.drop(columns=[target])

    # Split data
    test_threshold = test_threshold / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_threshold, random_state=42)
    st.write('X_train:', X_train.columns)
    st.write('y_train:', y_train.name)
    st.write('X_test:', X_test.columns)

    return X_train, X_test, y_train, y_test

def train_models(pipelines, X_train, X_test, y_train, y_test, target_type):
    """
    Train each model sequentially and store results in a DataFrame.
    """
    results = []

    for model_name, pipeline in pipelines.items():
        print(f"Training model: {model_name}")

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # Collect performance metrics
        if target_type == 'object':  # Classification
            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred, average="weighted"),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted")
            }
        else:  # Regression
            metrics = {
                "Model": model_name,
                "Mean Squared Error": mean_squared_error(y_test, y_pred),
                "Mean Absolute Error": mean_absolute_error(y_test, y_pred),
                "R2 Score": r2_score(y_test, y_pred),
                "Adjusted R2": adjusted_r2(r2_score(y_test, y_pred), X_test.shape[0], X_test.shape[1])
            }

        results.append(metrics)

    # Convert results list into a DataFrame
    results_df = pd.DataFrame(results)

    return results_df