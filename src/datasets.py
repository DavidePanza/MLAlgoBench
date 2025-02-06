import streamlit as st
import base64
import requests
import pandas as pd
import os



def initialize_session_state():
    params = st.query_params
    if "selected" in params:
        try:
            # Convert query parameter value to int and update session state.
            selected = int(params["selected"][0])
            # Only update if different from the current selection
            if st.session_state.get("selected_image") != selected:
                st.session_state.selected_image = selected
                st.session_state.df = None  # Reset df to load new data.
        except ValueError:
            pass  


def initialize_images():
    # Get the current script directory.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative paths to the image folder.
    image1_path = os.path.join(script_dir, "..", "image", "wine.jpeg")
    image2_path = os.path.join(script_dir, "..", "image", "titanic.jpeg")
    image3_path = os.path.join(script_dir, "..", "image", "iris.jpeg")

    # Define images and their corresponding CSV download URLs.
    images = [
        ("Wine Datset", image1_path, "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"),
        ("Titanic Dataset", image2_path, "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"),
        ("Iris Dataset", image3_path, "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"), 
    ]
    return images


def encode_image(image_path):
    """Read and encode an image to Base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def display_images(images):
    # Create five columns. The first and last columns will be empty, centering the three middle columns.
    cols = st.columns([1, 1, 1, 1, 1])

    # Loop over the images; use the middle columns (cols[1], cols[2], cols[3]).
    for i, (title, img_path, csv_url) in enumerate(images):
        # Use cols[i+1] so that for i==0 we use the second column, i==1 uses the third, etc.
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




def download_dataset():
    images = initialize_images()
    if st.session_state.get("selected_image") is not None:
        selected_index = st.session_state.selected_image
        title, img_path, csv_url = images[selected_index]
        
        # If not already loaded, fetch and store the CSV as a DataFrame.
        if st.session_state.df is None:
            response = requests.get(csv_url)
            if response.status_code == 200:
                df = pd.read_csv(csv_url)
                st.session_state.df = df
            else:
                st.error("Failed to download CSV data.")
        
        # Display the DataFrame if available.
        if st.session_state.df is not None:
            st.write(f"### Data from {title}")
            st.dataframe(st.session_state.df)
        
        return df
