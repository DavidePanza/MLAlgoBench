import streamlit as st
import pandas as pd
import os
import numpy as np
import base64
from data_loader import upload_files
from preprocessing import *
from data_tests import *
from feat_selection import *
from mypipeline import *
from mylogging import *
from models import *
from train_test import *
from datasets import *
from utils import *
from plots import *

def main():
    # Configure page
    configure_page()
    load_background_image()
    breaks(2)
    page_description()

    # Logging
    use_logging = False
    logger, log_stream = configure_logging() # Get logger and log stream
    if use_logging:
        logging_level = st.selectbox("Select logging level", ['INFO', 'DEBUG', 'WARNING'])
        toggle_logging(logging_level, logger)
    breaks(2)   

    # Upload user file
    st.markdown("<h3 style='text-align: left;padding-left: 5px;'>1.&nbsp;&nbsp;&nbsp;Data Loading</h3><br>",unsafe_allow_html=True)
    with st.container(height=700):
        breaks(1)
        st.write("Upload your data or experiment with one of the datasets provided:")
        df = upload_files()
        breaks(2)

        # Display toy dataset images and upload toy datasets
        if "selected_image" not in st.session_state:
            st.session_state.selected_image = None
        initialize_session_state()
        images = initialize_images()
        display_images(images)
        df = download_dataset()

        # If data is loaded, proceed
        if df is not None:
            st.session_state["data_loaded"] = True  
            logger.info(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns\n")
        else:
            st.stop()  # Stop execution if no data is uploaded
    breaks(1)

    # ---- Feature selection ----
    logger.info("------Starting feature selection process...\n")

    # Drop duplicates
    df = df.drop_duplicates()

    # Data visualization
    if df is not None:
        st.markdown("<h3 style='text-align: left;padding-left: 5px;'>2.&nbsp;&nbsp;&nbsp;Visual Data Exploration</h3><br>",unsafe_allow_html=True)
        with st.expander("Click here for data visualisation"):
            breaks(1)
            # Display data histograms
            col_arg1, _, col_arg2, _, _ = st.columns([1, .1, 1, 1, 1])
            with col_arg1:
                bins = st.number_input("Number os bins:", min_value=5, max_value=150, value=30)
            with col_arg2:
                col_x_row = st.number_input("Histogram per row:", min_value=2, max_value=6, value=5)
            plot_histograms(df,col_x_row,bins)
        
            # Display correlation matrix
            if df.select_dtypes(include=['number']).shape[1] > 1:
                breaks(1)
                if df.select_dtypes(include=['number']).shape[1] > 10:
                    matrix_size = 2.4
                else:
                    matrix_size = 1.6
                _, col_corr1, _, col_corr2, _, = st.columns([.15,.4, .1, matrix_size, .6])
                with col_corr1:
                    breaks(3)
                    zmin = st.number_input("Min value for correlation matrix:", min_value=-1, max_value=1, value=-1)
                    zmax = st.number_input("Max value for correlation matrix:", min_value=-1, max_value=1, value=1)
                with col_corr2:
                    plot_correlation_matrix(df.select_dtypes(include=['number']),zmin,zmax)
    breaks(1)

    # Target Selection
    breaks(1)
    st.markdown("<h3 style='text-align: left;padding-left: 5px;'>3.&nbsp;&nbsp;&nbsp;Target and Features Selection</h3><br>",unsafe_allow_html=True)
    feat_container_height = 350 if df.columns.shape[0] > 10 else 250
    with st.container(height = feat_container_height):
        breaks(1)
        _, col2, col3, col4, _ = st.columns([.2, 1, 1, 1, .2])
        with col4:
            target, target_type = select_target(df)
            logger.info(f"Target: {target}")

        # Feature Selection
        numeric_feat, categorical_feat = return_feat(df, target)
        selected_numeric, selected_categorical = select_vars(numeric_feat, categorical_feat, col2, col3)
        df = drop_columns(df, selected_numeric, selected_categorical, target)

        # Log selections
        logger.info(f"Numeric features: {' '.join(numeric_feat)}")
        logger.info(f"Categorical features: {' '.join(categorical_feat)}\n")
        logger.info(f"Selected numeric features: {' '.join(selected_numeric)}")
        logger.info(f"Selected categorical features: {' '.join(selected_categorical)}")
        logger.info(f"--> Columns in df: {' '.join(df.columns)}")
        logger.info(f"--> Rows in df: {df.shape[0]}")
        breaks(2)

        # Store selections and update session state
        st.session_state["target"] = target
        st.session_state["target_type"] = target_type
        st.session_state["selected_numeric"] = selected_numeric
        st.session_state["selected_categorical"] = selected_categorical

    breaks(1)
    st.markdown("<h3 style='text-align: left;padding-left: 5px;'>4.&nbsp;&nbsp;&nbsp;Data Preprocessing</h3><br>",unsafe_allow_html=True)
    with st.container(height = 300):
        breaks(1)

        # ---- Data Preprocessing ----
        logger.info("\n\n------Starting data preprocessing process\n")

        # Missing Values and Outliers
        _, col_p2, _, col_p4, _ = st.columns([.1, 1, .4, 1, .1])
        with col_p2:
            drop_cols_missing= st.radio("Do you want to drop columns with missing values?", ("Yes", "No"),  index=1)
            st.session_state["drop_cols_missing"] = drop_cols_missing
            if st.session_state["drop_cols_missing"] == "Yes":
                breaks(1)
                missing_threshold = st.slider("Select threshold of missing values to drop variables", 0, 100, 20)
        with col_p4:
            drop_cols_outliers = st.radio("Do you want to drop outliers?", ("Yes", "No"), index=1)
            st.session_state["drop_cols_outliers"] = drop_cols_outliers
            if st.session_state["drop_cols_outliers"] == "Yes": 
                breaks(1)
                outliers_threshold = st.slider("Select z-threshold to drop outliers", 0, 6, 3)

        # Drop cols with missing values
        if st.session_state["drop_cols_missing"] == "Yes":
            df, cols_to_drop = process_missing_values(df, selected_numeric, selected_categorical, logger, missing_threshold)
            selected_numeric = [col for col in selected_numeric if col not in cols_to_drop]
            selected_categorical = [col for col in selected_categorical if col not in cols_to_drop]
            logger.info(f"Columns dropped due to missing values: {', '.join(cols_to_drop)}")
        
        # Drop outliers
        if st.session_state["drop_cols_outliers"] == "Yes":
            df = process_outliers(df, target, outliers_threshold)
            logger.info(f"Remaining columns after drop: {' '.join(df.columns)}")
            logger.info(f"Target: {target}")
            logger.info(f"--> Columns in df: {' '.join(df.columns)}")

        # Drop NAs in target
        df = remove_target_na(df, target)
        logger.info(f"--> Rows in df: {df.shape[0]}")

        # Normality Test
        #selected_numeric = [col for col in selected_numeric if col not in cols_to_drop]
        normal_cols, not_normal_cols = run_shapiro_test(df, selected_numeric)
        logger.info(f"\n\t--Shapiro Test Results:")
        logger.info(f"Normal variables: {' '.join(normal_cols)}")
        logger.info(f"Not normal variables: {' '.join(not_normal_cols)}")

        # Cardinality
        #selected_categorical = [col for col in selected_categorical if col not in cols_to_drop]
        high_cardinality, low_cardinality = check_cardinality(df, selected_categorical)
        logger.info(f"\n\t--Cardinality Check Results:")
        logger.info(f"High cardinality variables: {' '.join(high_cardinality)}")
        logger.info(f"Low cardinality variables: {' '.join(low_cardinality)}")

    breaks(1)
    if st.button("Apply Preprocessing"):
        st.session_state["numeric_for_test"] = selected_numeric
        st.session_state["categorical_for_test"] = selected_categorical
        st.session_state["normal_distributed"] = normal_cols
        st.session_state["not_normal_distributed"] = not_normal_cols
        st.session_state["high_cardinality"] = high_cardinality
        st.session_state["low_cardinality"] = low_cardinality
        st.session_state["preprocessed"] = True
    breaks(1)

    if st.session_state.get("preprocessed", False):
        logger.info("\n\n------Model Selection\n")

        # Select Models
        breaks(1)
        st.markdown("<h3 style='text-align: left;padding-left: 5px;'>5.&nbsp;&nbsp;&nbsp;Training Configuration</h3><br>",unsafe_allow_html=True) 
        with st.container(height = 300):  
            breaks(1)
            if target_type == 'object':
                models = get_categorical_models()
            else:
                models = get_regression_models()

            _, col_m2, _, col_m4, _ = st.columns([.1, 1, .4, 1, .1])
            with col_m2:
                selected_models = model_selection(models)
            filtered_models = {model_name: model for model_name, model in models.items() if model_name in selected_models}
            logger.info(f"Selected models: {' '.join(selected_models)}")

            # prepare models pipeline
            models_pipelines = create_model_pipelines(filtered_models, st.session_state.normal_distributed, 
                                                    st.session_state.not_normal_distributed, st.session_state.low_cardinality, st.session_state.high_cardinality)
            logger.info(f"models pipelines: {models_pipelines.items()}")

            # Train and evaluate models
            with col_m4:
                test_threshold = st.slider("Select size of test set (%)", 0, 100, 20)
            logger.info(f"df columns: {df.keys(), df.shape[0]}")
            X_train, X_test, y_train, y_test = prepare_data(df, st.session_state.target, test_threshold)
            logger.info(f"Columns annd rows in df: {df.keys(), df.shape[0]}")
            logger.info(f"N° cols in train: {X_train.columns}")
            logger.info(f"N° cols in test: {y_train.to_frame().columns}")
            logger.info(f"train size: {X_train.shape[0]}, test_size:{X_test.shape[0]}")
            
        breaks(1)
        if st.button("Train Models"):
            results_df, metrics_name = train_models(models_pipelines, X_train, X_test, y_train, y_test, st.session_state.target_type)
            st.session_state["metrics_name"] = metrics_name
            st.session_state["results_df"] = results_df
            st.session_state["model trained"] = True
            #st.dataframe(results_df)
            breaks(1)

    # Results visualization
    if st.session_state.get("preprocessed", False) and st.session_state.get("model trained", False):
        st.markdown("<h1 style='text-align: center;'>6.&nbsp;&nbsp;Results</h1><br>", unsafe_allow_html=True)
        metric = st.selectbox("Select Metric", st.session_state.metrics_name)
        breaks(1)
        plot_results(st.session_state.results_df,  metric)

    # Display logs
    if use_logging:
        display_logs(log_stream)


if __name__ == "__main__":
    main()

