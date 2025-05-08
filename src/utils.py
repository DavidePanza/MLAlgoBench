import streamlit as st
import base64

def configure_page() -> None:
    """
    Configures the Streamlit page.
    """
    st.set_page_config(page_title="ML Algo Benchmarker", layout="wide")

def reset_session_state():
    """
    Resets all session state variables.
    """
    st.session_state.clear()  

def separation():
    """
    Creates a separation line.
    """
    st.write("\n")
    st.markdown("---")

def breaks(n=1):
    """
    Creates a line break.
    """
    if n == 1:
        st.markdown("<br>",unsafe_allow_html=True)
    elif n == 2:
        st.markdown("<br><br>",unsafe_allow_html=True)
    elif n == 3:
        st.markdown("<br><br><br>",unsafe_allow_html=True)
    else:
        st.markdown("<br><br><br><br>",unsafe_allow_html=True)

def get_base64_encoded_image(image_path):
    """
    Reads an image file and encodes it to Base64.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def load_background_image():
    """
    Loads and displays a background image with an overlaid title.
    """
    image_path = "../images/background4.jpeg"  
    base64_image = get_base64_encoded_image(image_path)
    
    # Inject CSS for the background and title overlay
    st.markdown(
        f"""
        <style>
        /* Background container with image */
        .bg-container {{
            position: relative;
            background-image: url("data:image/png;base64,{base64_image}");
            background-size: cover;
            background-position: 50% 15%;
            height: 400px;  /* Adjust the height of the background */
            width: 100%;
            filter: brightness(135%); /* Dim the brightness of the image */
            border-radius: 200px;  /* Makes the container's corners rounded */
            overflow: hidden;  
        }}

        /* Overlay for dimming effect */
        .bg-container::after {{
            content: '';
            position: absolute;
            top: ;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(20, 20, 20, 0.5); /* Semi-transparent black overlay */
            z-index: 1; /* Ensure the overlay is above the image */
        }}

        /* Overlay title styling */
        .overlay-title {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;   /* Title color */
            font-size: 70px;
            font-weight: bold;
            text-shadow: 2px 2px 8px rgba(0, 0, 0, 0.7); /* Shadow for better visibility */
            text-align: center;
            z-index: 2; /* Ensure the title is above the overlay */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create the background container with an overlaid title
    st.markdown(
        """
        <div class="bg-container">
            <div class="overlay-title">ML Models Benchmarker</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def page_description():
    """
    Displays a description of the app's functionality with numbered steps.
    """
    steps = [
        "<strong>Load Data</strong> – Upload your own CSV or Excel file, or explore preloaded datasets.",
        "<strong>Explore Data</strong> – Visualize distributions, check correlations, and inspect data quality.",
        "<strong>Select Target & Features</strong> – Choose your prediction target and relevant input variables.",
        "<strong>Preprocess</strong> – Handle missing values, remove outliers, test normality, and manage categorical features.",
        "<strong>Train Models</strong> – Choose from classification or regression algorithms and configure test/train splits.",
        "<strong>Compare Performance</strong> – Benchmark model performance using multiple evaluation metrics and interactive plots.",
    ]

    html = "<p style='font-size: 16px;'>This app helps you build and evaluate machine learning models step by step. You can:</p><ol style='font-size: 16px;'>"
    for step in steps:
        html += f"<li>{step}</li>"
    html += "</ol>"

    st.markdown(html, unsafe_allow_html=True)

