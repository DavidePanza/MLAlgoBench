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
            logger.info(f'Features with missing values: {col} {avg_null:.0f}%') 
            if avg_null > treshold:
                cols_to_drop.append(col)

    return df.drop(columns=cols_to_drop), cols_to_drop

def process_outliers(df, target, threshold=2):
    if isinstance(df, pd.Series):
        # Compute the z-score for the Series
        z_scores = zscore(df)
        # Find indices where the absolute z-score is greater than threshold
        outlier_idx = df.index[np.abs(z_scores) > threshold]
        return df.drop(outlier_idx)

    # If input is a DataFrame:
    elif isinstance(df, pd.DataFrame):
        # Select only numeric columns
        num_cols = df.select_dtypes(include='number')
        # If a target column is provided and exists, drop it from the columns to check.
        if target is not None and target in num_cols.columns:
            num_cols = num_cols.drop(columns=[target], errors='ignore')
        
        # Compute z-scores for each numeric column
        z_scores_df = num_cols.apply(zscore)
        # Create a boolean mask: True where the absolute z-score is above threshold
        outlier_mask = np.abs(z_scores_df) > threshold
        # Identify rows where any numeric column is an outlier
        rows_to_remove = df.index[outlier_mask.any(axis=1)]
        # Drop those rows from the original DataFrame
        return df.drop(rows_to_remove)
    
    else:
        raise ValueError("Input must be a Pandas DataFrame or Series")

def remove_target_na(df, target):
    df = df.dropna(subset=[target])

    return df