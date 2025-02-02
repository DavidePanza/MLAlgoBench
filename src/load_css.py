import streamlit as st
import os

# Load custom CSS from an external file
def load_css():
    css_file_path = os.path.join("assets", "css", "style.css") 
    with open(css_file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
