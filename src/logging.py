import streamlit as st
import logging
import io

# Configure logging
def configure_logging():
    log_stream = io.StringIO()  # Create a StringIO buffer to capture logs
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger, log_stream

# Function to dynamically set logging level
def toggle_logging(level, logger):
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'WARNING':
        logger.setLevel(logging.WARNING)
    else:
        logger.warning(f"Unknown logging level: {level}. Using INFO as default.")
        logger.setLevel(logging.INFO)


# Show logs in the Streamlit app
def display_logs(log_stream):
    log_stream.seek(0)  # Go to the start of the StringIO buffer
    logs = log_stream.read()  # Read the captured log
    st.text(logs)  # Display the logs in Streamlit app