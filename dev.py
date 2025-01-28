import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split


def separation():
    st.write("\n\n")
    st.markdown("---")

def configure_page() -> None:
    # metadata
    st.set_page_config(page_title="ML Algo Benchmarker", layout="wide")


def configure_overview() -> None:
    st.markdown("## Overview")
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
        if "df_clean" in st.session_state:
            del st.session_state.df_clean  # Clear the cleaned DataFrame

        # Read the CSV file into a DataFrame
        try:
            df = pd.read_csv(uploaded_file)
            # Display the DataFrame
            return df
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return None
    return None


def return_feat(df: pd.DataFrame) -> list:
    numeric_cols = df.select_dtypes(include='number')
    categorical_cols = df.select_dtypes(include='object')
    numeric_feat = numeric_cols.columns.to_list()
    categorical_feat = categorical_cols.columns.to_list()

    return numeric_feat, categorical_feat


def select_vars(numeric_feat, categorical_feat):
    # Store selected states
    selected_numeric = []
    selected_categorical = []

    # Create two columns
    col1, col2 = st.columns(2)

    # Display numerical options in the first column
    with col1:
        st.subheader("Numerical Variables")
        for var in numeric_feat:
            # Checkbox for each variable
            selected = st.checkbox(var, value=(var in selected_numeric))
            if selected and var not in selected_numeric:
                selected_numeric.append(var)
            elif not selected and var in selected_numeric:
                selected_numeric.remove(var)

    with col2:
        st.subheader("Categorical Variables")
        for var in categorical_feat:
            # Checkbox for each variable
            selected = st.checkbox(var, value=(var in selected_categorical))
            if selected and var not in selected_categorical:
                selected_categorical.append(var)
            elif not selected and var in selected_categorical:
                selected_categorical.remove(var)
    
    st.write("Selected Items:", selected_numeric)
    st.write("Selected Items:", selected_categorical)

    return selected_numeric, selected_categorical


def process_missing_values(df: pd.DataFrame, selected_numeric, selected_categorical, treshold = .2) -> pd.DataFrame:
    # drop columns with missing values
    cols_to_drop = []
    df_selected = df[selected_numeric + selected_categorical]
    print(df_selected)

    for col in df_selected.columns: 
        len_df = df_selected.shape[0]
        if sum(df_selected[col].isnull()) > 0:
            tot_null = sum(df_selected[col].isnull())
            avg_null = (tot_null/len_df) * 100
            st.write(f'{col:20} % null values: {avg_null}%')
            if avg_null > treshold:
                cols_to_drop.append(col)

    st.write(cols_to_drop)

    return df.drop(columns=cols_to_drop), cols_to_drop


def process_outliers(df: pd.DataFrame, treshold: int = 2) -> pd.DataFrame:
    num_cols = df.select_dtypes(include='number')
    df_z = num_cols.apply(zscore)
    outliers = abs(df_z) > treshold
    row_to_remove = np.where(outliers.any(axis=1))[0]
    df = df.drop(row_to_remove)

    return df


def main() -> None:
    configure_page()
    configure_overview()
    separation()

    df = upload_files()
    separation()

    if df is not None:
        if "df_clean" not in st.session_state:
            st.session_state.df_clean = df

        numeric_feat, categorical_feat = return_feat(st.session_state.df_clean)
        st.markdown("# Features Selection")
        selected_numeric, selected_categorical = select_vars(numeric_feat, categorical_feat)
        separation()

        st.markdown("# Data Preprocessing")
<<<<<<< HEAD
        st.markdown("<br><h4>1 Drop Columns with Missing Values</43>", unsafe_allow_html=True)
        na_threshold = st.slider("Select threshold of missing values to drop variables", 0, 100, 20)
        st.markdown("<br><h4>2 Remove Outliers</h4>", unsafe_allow_html=True)
=======
        st.markdown("<br><h4>1 Drop Columns with Missing Values</h4>", unsafe_allow_html=True)
        na_threshold = st.slider("Select threshold of missing values to drop variables", 0, 100, 20)
        st.markdown("<br><h4>2 Remove Outliers</h4>", unsafe_allow_html=True)
>>>>>>> vars_to_list
        outliers_threshold = st.slider("Select z-threshold to drop outliers", 0, 6, 3)

        # Button to trigger column drop
        if st.button("Apply Preprocessing"):
            rows_before = st.session_state.df_clean.shape[0]
            st.session_state.df_clean, cols_to_drop = process_missing_values(
                st.session_state.df_clean, selected_numeric, selected_categorical, na_threshold)
            st.session_state.df_clean = process_outliers(st.session_state.df_clean, outliers_threshold)
            st.session_state.df_clean = st.session_state.df_clean.drop_duplicates()
            st.write(f"Remaining columns after drop: {st.session_state.df_clean.columns}")
            st.write(f"rows before: {rows_before}, rows after: {st.session_state.df_clean.shape[0]}")
    

if __name__ == "__main__":
    main()
