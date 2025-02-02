import streamlit as st
import pandas as pd
import os
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import category_encoders as ce
from src.tests import *
from src.feat_selection import *
from src.pipeline import *
from src.logging import *
from src.models import *


def separation():
    st.write("\n")
    st.markdown("---")


def configure_page() -> None:
    # metadata
    st.set_page_config(page_title="ML Algo Benchmarker", layout="wide")


def configure_overview() -> None:
    st.markdown(
    "<h1 style='text-align: center;'>Overview</h1>",
    unsafe_allow_html=True
    )
    st.write('\n\n')
    st.markdown(
        "This app compares the performance of different machine learning algorithms."
    )


def upload_files() -> pd.DataFrame:
    """Function to upload CSV files and return the DataFrame."""
    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    # Check if the user uploaded a file
    if uploaded_file is not None:
        # Reset session state to clear previous transformations
        st.session_state.clear()

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            # Display the DataFrame
            return df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return None
    return None


def process_missing_values(df: pd.DataFrame, selected_numeric, selected_categorical, logger, treshold = .2) -> pd.DataFrame:
    # drop columns with missing values
    cols_to_drop = []
    df_selected = df[selected_numeric + selected_categorical]

    for col in df_selected.columns: 
        len_df = df_selected.shape[0]
        if sum(df_selected[col].isnull()) > 0:
            tot_null = sum(df_selected[col].isnull())
            avg_null = (tot_null/len_df) * 100
            logger.info(f'{col} % null values: {avg_null:.2f}') 
            if avg_null > treshold:
                cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop


def process_outliers(df: pd.DataFrame, target, treshold: int = 2) -> pd.DataFrame:
    num_cols = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
    df_z = num_cols.apply(zscore)
    outliers = abs(df_z) > treshold
    row_to_remove = np.where(outliers.any(axis=1))[0]
    df = df.drop(row_to_remove)

    return df


def remove_target_na(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.dropna(subset=[target])

    return df


# ---- RUN MAIN ----

def main() -> None:
    configure_page()
    configure_overview()
    logger, log_stream = configure_logging()  # Get logger and log stream
    logging_level = st.selectbox("Select logging level", ['INFO', 'DEBUG', 'WARNING'])
    toggle_logging(logging_level, logger)

    df = upload_files()
    separation()

    if df is not None:
        if "df_clean" not in st.session_state:
            st.session_state.df_clean = df
        
        logger.info(f"Vars stored: {' '.join(st.session_state.df_clean.columns)}")
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = [] 

        # Target Selection
        st.markdown(
        "<h1 style='text-align: center;'>Target and Features Selection</h1><br><br>",
        unsafe_allow_html=True
        )
        target, target_type = select_target(st.session_state.df_clean)

        # Feature Selection
        numeric_feat, categorical_feat = return_feat(st.session_state.df_clean, target)
        st.markdown("<br><h3> - Select Features:</h3>", unsafe_allow_html=True)
        selected_numeric, selected_categorical = select_vars(numeric_feat, categorical_feat)
        logger.info(f"Selected numeric: {' '.join(selected_numeric)}")
        logger.info(f"Selected categorical: {' '.join(selected_categorical)}")
        separation()
        
        # Data Preprocessing 1 - Missing Values and Outliers
        st.markdown("<h1 style='text-align: center;'>Data Preprocessing</h1><br>", unsafe_allow_html=True)
        st.markdown("<br><h3>1 Drop Columns with Missing Values</h3>", unsafe_allow_html=True)
        na_threshold = st.slider("Select threshold of missing values to drop variables", 0, 100, 20)
        st.markdown("<br><h3>2 Remove Outliers</h3>", unsafe_allow_html=True)
        outliers_threshold = st.slider("Select z-threshold to drop outliers", 0, 6, 3)
        
        if 'preprocessed' not in st.session_state:
            st.session_state.preprocessed = False
        
        if st.button("Apply Preprocessing"):
            st.session_state.preprocessed = False

            logger.info(f"\t\tStarting Preprocessing:")
            logger.info(f"Rows before preprocessing: {st.session_state.df_clean.shape[0]}")
            
            # Drop cols with missing values
            st.session_state.df_clean, cols_to_drop = process_missing_values(
                st.session_state.df_clean, selected_numeric, selected_categorical, logger, na_threshold)
            logger.info(f"Columns dropped due to missing values: {', '.join(cols_to_drop)}")
            
            # Drop outliers
            st.session_state.df_clean = process_outliers(st.session_state.df_clean, target, outliers_threshold)

            # Drop duplicates
            st.session_state.df_clean = st.session_state.df_clean.drop_duplicates()
            logger.info(f"Remaining columns after drop: {' '.join(st.session_state.df_clean.columns)}")
            logger.info(f"Target: {target}")
            logger.info(f"Variables: {' '.join(st.session_state.df_clean.keys())}")

            # Drop NAs in target
            st.session_state.df_clean = remove_target_na(st.session_state.df_clean, target)
            logger.info(f"Remaining rows after NA drop in target: {st.session_state.df_clean.shape[0]}")

            # Data Preprocessing 2 - Normality Test - Cardinality  

            # Normality Test
            selected_numeric = [col for col in selected_numeric if col not in cols_to_drop]
            normal_cols, not_normal_cols = run_shapiro_test(st.session_state.df_clean, selected_numeric)
            logger.info(f"\t\tShapiro Test Results:")
            logger.info(f"Normal variables: {' '.join(normal_cols)}")
            logger.info(f"Not normal variables: {' '.join(not_normal_cols)}")

            # Cardinality
            selected_categorical = [col for col in selected_categorical if col not in cols_to_drop]
            high_cardinality, low_cardinality = check_cardinality(st.session_state.df_clean, selected_categorical)
            st.session_state.preprocessed = True

        if st.session_state.preprocessed:

            # Initialize pipeline
            preprocessor = create_model_pipeline(
                numeric_normal_features=normal_cols, 
                numeric_not_normal_features=not_normal_cols, 
                categorical_onehot=low_cardinality, 
                categorical_binary=high_cardinality
            )
            logger.info(f"Pipeline structure: {preprocessor}")

            # Select Models
            separation()
            st.markdown("<h1 style='text-align: center;'>Model Selection</h1><br>", unsafe_allow_html=True)    
            if target_type == 'object':
                models = get_categorical_models()
            else:
                models = get_regression_models()
            #selected_models = model_selection(models)
            selected_models = model_selection(models)
            logger.info(f"Selected models: {' '.join(selected_models)}")

            # Display logs
            display_logs(log_stream)


if __name__ == "__main__":
    main()
