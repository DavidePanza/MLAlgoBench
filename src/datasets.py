import streamlit as st
import base64
import requests
import pandas as pd
import os

def initialize_session_state():
    """
    Initialize session state variables.
    """
    params = st.query_params
    if "selected" in params:
        try:
            # Convert query parameter value to int and update session state.
            selected = int(params["selected"][0])
            # Only update if different from the current selection
            if st.session_state.get("selected_image") != selected:
                st.session_state.selected_image = selected
                st.session_state.df = None  
        except ValueError:
            pass  

def initialize_images():
    """
    Initialize images for the predefined datasets and their corresponding CSV download URLs.
    """
    # Get the directory of the current script and define the paths to the images.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image1_path = os.path.join(script_dir, "..", "images", "wine.jpeg")
    image2_path = os.path.join(script_dir, "..", "images", "titanic.jpeg")
    image3_path = os.path.join(script_dir, "..", "images", "iris.jpeg")

    # Define images and their corresponding CSV download URLs.
    images = [
        ("Wine Datset", image1_path, "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),
        ("Titanic Dataset", image2_path, "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"),
        ("Iris Dataset", image3_path, "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"), 
    ]
    return images

def encode_image(image_path):
    """
    Read and encode an image to Base64.
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def display_images(images):
    """
    Display images with titles and links to download the corresponding CSV files.
    """
    cols = st.columns([1, 1, 1, 1, 1])

    for i, (title, img_path, csv_url) in enumerate(images):
        with cols[i + 1]:
            st.markdown(f"<h3 style='text-align: center;'>{title}</h3>", unsafe_allow_html=True)
            if os.path.exists(img_path):
                img_base64 = encode_image(img_path)
                border_style = (
                    "3px solid red" if st.session_state.get("selected_image") == i 
                    else "3px solid transparent"
                )
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <a href="?selected={i}" target="_self">
                            <div style="
                                display: inline-block;
                                border: {border_style};
                                padding: 5px;
                                border-radius: 150px;
                                box-sizing: border-box;
                            ">
                                <img src="data:image/jpeg;base64,{img_base64}" 
                                    style="display: block; width:100%; border-radius: inherit; cursor: pointer;">
                            </div>
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

def download_dataset(show_data=False):
    """
    Download the selected dataset and display its DataFrame.
    """
    df = None 
    images = initialize_images()
    if st.session_state.get("selected_image") is not None:
        selected_index = st.session_state.selected_image
        title, img_path, csv_url = images[selected_index]
        
        # Check if the session state DataFrame is already set.
        if st.session_state.df is None:
            response = requests.get(csv_url)
            if response.status_code == 200:
                if csv_url == "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv":
                    df = pd.read_csv(csv_url, sep=";")
                else:
                    df = pd.read_csv(csv_url)
                st.session_state.df = df  
            else:
                st.error("Failed to download CSV data.")
        else:
            None

        # Display the DataFrame if available and show_data is set to True.
        if df is not None and show_data:
            st.write(f"### Data from {title}")
            st.dataframe(df)
    
        return st.session_state.df
    
    else:
        return None


