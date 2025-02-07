import pandas as pd
import streamlit as st

def select_target(df):
    st.markdown("<h5 style='text-align:left;'>Target Variable:</h5>", unsafe_allow_html=True)
    target = st.selectbox("", df.columns,label_visibility="collapsed")
    target_type = df[target].dtypes

    return target, target_type

def return_feat(df, target):
    numeric_cols = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
    categorical_cols = df.select_dtypes(include='object').drop(columns=[target], errors='ignore')
    numeric_feat = numeric_cols.columns.to_list()
    categorical_feat = categorical_cols.columns.to_list()

    return numeric_feat, categorical_feat

def select_vars(numeric_feat, categorical_feat,col1, col2):
    """Checkbox selection for features."""
    
    # Displaying in two columns
    #col1, col2 = st.columns(2)

    # Numerical variables selection in the first column
    with col1:
        st.markdown("<h5 style='text-align:left;'>Numerical Variables:</h5>", unsafe_allow_html=True)
        selected_numeric = [var for var in numeric_feat if st.checkbox(var, key=f"num_{var}")]

    # Categorical variables selection in the second column
    with col2:
        st.markdown("<h5 style='text-align:left;'>Categorical Variables:</h5>", unsafe_allow_html=True)
        selected_categorical = [var for var in categorical_feat if st.checkbox(var, key=f"cat_{var}")]

    return selected_numeric, selected_categorical

def drop_columns(df, selected_numeric, selected_categorical, target):
    """Drop columns from DataFrame."""
    cols_to_drop = [col for col in df.columns if col not in selected_numeric + selected_categorical + [target]]
    df = df.drop(columns=cols_to_drop)
    return df