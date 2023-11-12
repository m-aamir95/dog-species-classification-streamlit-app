"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st

import os

# Set page configuration to wide mode
# st.set_page_config(layout="wide")

st.title("Dog Species Classification Via Convolutional Neural Networks")
# Section 1: Introduction
st.header("How it works")
st.write("##### The app is leveraging a custom built CNN model for classifying dog species. "
         "The model has been trained on the [Stanford dog breeds dataset](http://vision.stanford.edu/aditya86/ImageNetDogs). To achieve the best results, use images where the dog is easily distinguishable.")

# Section 2: Samples Dog images
st.header("Samples Images")
# Draw multiple images
sample_dogs_root_dir = "./Sample_Dogs"
# Generate HTML and CSS for the images

#Load and render sample dogs
for image_path in os.listdir(sample_dogs_root_dir):
    sample_dog_full_path = os.path.join(sample_dogs_root_dir, image_path)
    # html_content += f"<img src='{sample_dog_full_path}' alt='Image' class='image'>"
    st.image(sample_dog_full_path, "Beagle", width=400)

st.header("Upload")
# File uploader widget
uploaded_file = st.file_uploader("Choose an image...")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=400)

# Section 3: Conclusion
st.header("Conclusion")
st.write("##### Summarize your findings and conclude your Streamlit app.")

