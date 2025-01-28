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
    if "selected_numeric" not in st.session_state:
        st.session_state.selected_numeric = []
    if "selected_categorical" not in st.session_state:
        st.session_state.selected_categorical = []

    # Create two columns
    col1, col2 = st.columns(2)

    # Display numerical options in the first column
    with col1:
        st.subheader("Numerical Variables")
        for var in numeric_feat:
            # Checkbox for each variable
            selected = st.checkbox(var, value=(var in st.session_state.selected_numeric))
            if selected and var not in st.session_state.selected_numeric:
                st.session_state.selected_numeric.append(var)
            elif not selected and var in st.session_state.selected_numeric:
                st.session_state.selected_numeric.remove(var)

    with col2:
        st.subheader("Categorical Variables")
        for var in categorical_feat:
            # Checkbox for each variable
            selected = st.checkbox(var, value=(var in st.session_state.selected_categorical))
            if selected and var not in st.session_state.selected_categorical:
                st.session_state.selected_categorical.append(var)
            elif not selected and var in st.session_state.selected_categorical:
                st.session_state.selected_categorical.remove(var)
    
    st.write("Selected Items:", st.session_state.selected_numeric)
    st.write("Selected Items:", st.session_state.selected_categorical)


def process_na(df: pd.DataFrame, treshold=.20) -> pd.DataFrame:

    # drop columns with missing values
    cols_to_drop = []

    for col in df.columns: 
        len_df = df.shape[0]
        if sum(df[col].isnull()) > 0:
            tot_null = sum(df[col].isnull())
            avg_null = tot_null/len_df
            st.write(f'{col:20} % null values: {avg_null}%')
            if avg_null > treshold:
                cols_to_drop.append(col)

    st.write(cols_to_drop)
    # df = df.drop(columns=cols_to_drop)
    # df = df.drop(['Id'], axis=1)


def main() -> None:
    configure_page()
    configure_overview()
    separation()
    df = upload_files()
    separation()
    if df is not None:
        numeric_feat, categorical_feat = return_feat(df)
        select_vars(numeric_feat, categorical_feat)
        separation()
        process_na(df)
    


if __name__ == "__main__":
    main()
