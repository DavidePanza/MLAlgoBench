import streamlit as st
import pandas as pd
import os
import numpy as np
from src.preprocessing import *
from src.tests import *
from src.feat_selection import *
from src.pipeline import *
from src.logging import *
from src.models import *
from src.train_test import *


def separation():
    st.write("\n")
    st.markdown("---")

def breaks(n=1):
    if n == 1:
        st.markdown("<br>",unsafe_allow_html=True)
    elif n == 2:
        st.markdown("<br><br>",unsafe_allow_html=True)
    elif n == 3:
        st.markdown("<br><br><br>",unsafe_allow_html=True)
    else:
        st.markdown("<br><br><br><br>",unsafe_allow_html=True)

def configure_page() -> None:
    # metadata
    st.set_page_config(page_title="ML Algo Benchmarker", layout="wide")


def configure_overview() -> None:
    st.markdown(
    "<h1 style='text-align: center;'>Overview</h1>",
    unsafe_allow_html=True
    )
    breaks(2)
    st.markdown(
        "This app compares the performance of different machine learning algorithms."
    )

def reset_session_state():
    """Resets all session state variables."""
    st.session_state.clear()  # Clears everything

def upload_files():
    """Upload a CSV file and return it as a DataFrame."""
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # If no file is uploaded, reset everything
    if uploaded_file is None:
        reset_session_state()
        return None
    
    return pd.read_csv(uploaded_file)  # Read and return the DataFrame

def main():
    configure_page()
    configure_overview()
    breaks(1)
    logger, log_stream = configure_logging()  # Get logger and log stream
    logging_level = st.selectbox("Select logging level", ['INFO', 'DEBUG', 'WARNING'])
    toggle_logging(logging_level, logger)

    # Step 1: Upload file
    df = upload_files()
    logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns\n")
    separation()
    # If data is loaded, proceed
    if df is not None:
        st.session_state["data_loaded"] = True  # Mark that data is loaded
    else:
        st.stop()  # Stop execution if no data is uploaded

    # Step 2: Feature selection
    logger.info("------Starting feature selection process...\n")

    # Target Selection
    st.markdown("<h1 style='text-align: center;'>Target and Features Selection</h1><br><br>",unsafe_allow_html=True)
    target, target_type = select_target(df)
    logger.info(f"Target: {target}")

    # Feature Selection
    st.markdown("<br><h3> - Select Features:</h3>", unsafe_allow_html=True)
    numeric_feat, categorical_feat = return_feat(df, target)
    selected_numeric, selected_categorical = select_vars(numeric_feat, categorical_feat)
    df = drop_columns(df, selected_numeric, selected_categorical, target)

    logger.info(f"Numeric features: {' '.join(numeric_feat)}")
    logger.info(f"Categorical features: {' '.join(categorical_feat)}\n")
    logger.info(f"Selected numeric features: {' '.join(selected_numeric)}")
    logger.info(f"Selected categorical features: {' '.join(selected_categorical)}")
    logger.info(f"--> Remaining columns after drop (in df): {' '.join(df.columns)}")

    # Button to apply preprocessing
    breaks(2)
    if st.button("Apply Selection"):
        # Store selections and update session state
        st.session_state["target"] = target
        st.session_state["target_type"] = target_type
        st.session_state["selected_numeric"] = selected_numeric
        st.session_state["selected_categorical"] = selected_categorical
        st.session_state["feature selection"] = True  # Set preprocessing flag
    separation()

    # Step 3: Only show model selection **after** preprocessing is applied
    if st.session_state.get("feature selection", False):
        logger.info("\n\n------Starting data preprocessing process\n")

        # Data Preprocessing 1 - Missing Values and Outliers
        st.markdown("<h1 style='text-align: center;'>Data Preprocessing</h1><br>", unsafe_allow_html=True)
        st.markdown("<br><h3>1 Drop Columns with Missing Values</h3>", unsafe_allow_html=True)
        na_threshold = st.slider("Select threshold of missing values to drop variables", 0, 100, 20)
        st.markdown("<br><h3>2 Remove Outliers</h3>", unsafe_allow_html=True)
        outliers_threshold = st.slider("Select z-threshold to drop outliers", 0, 6, 3)

        # Drop cols with missing values
        df, cols_to_drop = process_missing_values(df, selected_numeric, selected_categorical, logger, na_threshold)
        logger.info(f"Columns dropped due to missing values: {', '.join(cols_to_drop)}")
        
        # Drop outliers
        df = process_outliers(df, target, outliers_threshold)

        # Drop duplicates
        df = df.drop_duplicates()
        logger.info(f"Remaining columns after drop: {' '.join(df.columns)}")
        logger.info(f"Target: {target}")
        logger.info(f"--> Columns in df: {' '.join(df.columns)}")

        # Drop NAs in target
        df = remove_target_na(df, target)
        logger.info(f"--> Rows in df: {df.shape[0]}")

        # Data Preprocessing 2 - Normality Test - Cardinality  
        # Normality Test
        selected_numeric = [col for col in selected_numeric if col not in cols_to_drop]
        normal_cols, not_normal_cols = run_shapiro_test(df, selected_numeric)
        logger.info(f"\n\t--Shapiro Test Results:")
        logger.info(f"Normal variables: {' '.join(normal_cols)}")
        logger.info(f"Not normal variables: {' '.join(not_normal_cols)}")

        # Cardinality
        selected_categorical = [col for col in selected_categorical if col not in cols_to_drop]
        high_cardinality, low_cardinality = check_cardinality(df, selected_categorical)
        logger.info(f"\n\t--Cardinality Check Results:")
        logger.info(f"High cardinality variables: {' '.join(high_cardinality)}")
        logger.info(f"Low cardinality variables: {' '.join(low_cardinality)}")

        breaks(2)
        if st.button("Apply Preprocessing"):
            st.session_state["numeric_for_test"] = selected_numeric
            st.session_state["categorical_for_test"] = selected_categorical
            st.session_state["normal_distributed"] = normal_cols
            st.session_state["not_normal_distributed"] = not_normal_cols
            st.session_state["high_cardinality"] = high_cardinality
            st.session_state["low_cardinality"] = low_cardinality
            st.session_state["preprocessed"] = True

    if st.session_state.get("feature selection", False) and st.session_state.get("preprocessed", False):
        logger.info("\n\n------Model Selection\n")

        # Select Models
        separation()
        st.markdown("<h1 style='text-align: center;'>Model Selection</h1><br>", unsafe_allow_html=True)    
        if target_type == 'object':
            models = get_categorical_models()
        else:
            models = get_regression_models()
        selected_models = model_selection(models)
        filtered_models = {model_name: model for model_name, model in models.items() if model_name in selected_models}
        logger.info(f"Selected models: {' '.join(selected_models)}")

        # prepare models pipeline
        models_pipelines = create_model_pipelines(filtered_models, st.session_state.normal_distributed, 
                                                st.session_state.not_normal_distributed, st.session_state.low_cardinality, st.session_state.high_cardinality)
        logger.info(f"models pipelines: {models_pipelines.items()}")

        breaks(2)
        if st.button("Select Models"):
            st.session_state["selected_models"] = selected_models
            st.session_state["model selected"] = True

    if st.session_state.get("feature selection", False) and st.session_state.get("preprocessed", False) and st.session_state.get("model selected", False):

        # Train and evaluate models
        separation()
        st.markdown("<h1 style='text-align: center;'>Model Training</h1><br>", unsafe_allow_html=True)
        test_threshold = st.slider("Select size of test set (%)", 0, 100, 20)
        logger.info(f"Target: {st.session_state.target}")
        logger.info(f"df columns: {df.keys(), df.shape[0]}")
        X_train, X_test, y_train, y_test = prepare_data(df, st.session_state.target, test_threshold)
        logger.info(f"train size: {X_train.shape[0], y_train.shape[0]}")
        
        breaks(2)
        if st.button("Train Models"):
            results_df = train_models(models_pipelines, X_train, X_test, y_train, y_test, st.session_state.target_type)
            st.write(results_df)

    # Display logs
    display_logs(log_stream)



if __name__ == "__main__":
    main()
