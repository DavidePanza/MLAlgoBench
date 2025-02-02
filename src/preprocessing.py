import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import zscore

def process_missing_values(df, selected_numeric, selected_categorical, logger, treshold = .2):
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

def process_outliers(df, target, treshold=2):
    num_cols = df.select_dtypes(include='number').drop(columns=[target], errors='ignore')
    df_z = num_cols.apply(zscore)
    outliers = abs(df_z) > treshold
    row_to_remove = np.where(outliers.any(axis=1))[0]
    df = df.drop(row_to_remove)

    return df

def remove_target_na(df, target):
    df = df.dropna(subset=[target])

    return df