import csv
import pandas as pd
import streamlit as st

def upload_files():
    """
    Upload a file and return a DataFrame.
    """
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == "csv":
            # Read a small sample to guess the delimiter
            sample = uploaded_file.read(1024).decode('utf-8')
            uploaded_file.seek(0)  
            
            try:
                dialect = csv.Sniffer().sniff(sample)
                delimiter = dialect.delimiter
            except csv.Error:
                # If the sniffer fails, default to comma
                delimiter = ','
            
            df = pd.read_csv(uploaded_file, sep=delimiter)
        
        elif file_extension in ["xlsx", "xls"]:
            df = pd.read_excel(uploaded_file)
        
        else:
            st.error("Unsupported file type.")
            return None
        
        st.session_state.df = df  
        st.session_state.data_source = "upload"  # (optional) tag the source if needed
        return df
    else:
        return None