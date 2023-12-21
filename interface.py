"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import numpy as np
import cv2
import os

from inference import do_the_complete_classification
from inference import load_classNames

# Set page configuration to wide mode
# st.set_page_config(layout="wide")

st.title("Dog Species Classification")
# Section 1: Introduction
st.header("How it works")
st.write("##### The app is leveraging a fine-tuned inception_v3 model for classifying 120 dog species. "
         "The model has been trained on the [Stanford dog breeds dataset](http://vision.stanford.edu/aditya86/ImageNetDogs). To achieve the best results, use images where the dog is easily distinguishable.")

st.write("##### Explore dog breeds with the dropdown menu.")
selected_option = st.selectbox("Select an option:", load_classNames(), index=0)

#We have reformat selected option so that it matches the naming structure of downloaded dog images
selected_option = '_'.join(selected_option.split(' '))
all_dog_breeds_names_with_serial_nums = os.listdir("./Raw_Data/Images")

# Go through the selected dog breed dir and select a random dog to display
random_dog_img_path_from_the_selected_dog_breed = ""
for dog_breeds_dir in all_dog_breeds_names_with_serial_nums:

    if dog_breeds_dir.split('-')[1].lower().strip() == selected_option.lower().strip():
        selected_breed_all_available_dogs = os.listdir(os.path.join("./Raw_Data/Images", dog_breeds_dir))
        random_dog_name = selected_breed_all_available_dogs[np.random.randint(0, len(selected_breed_all_available_dogs)-1)]
        random_dog_img_path_from_the_selected_dog_breed = os.path.join("./Raw_Data/Images", dog_breeds_dir, random_dog_name)


# Section 2: Samples Dog images
st.header("Samples Images")

#Display a random picture of dog from the selected breed
st.image(random_dog_img_path_from_the_selected_dog_breed, selected_option, width=400)

# Get a prediction from the model


st.header("Upload")
# File uploader widget
uploaded_file = st.file_uploader("Choose an image...")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    # Convert the read image to numpy array 
    np_1d_buffer = np.frombuffer(uploaded_file.read(),np.uint8)

    # The data is still in the numpy format, we need to decode it via opencv decode function which
    # Will convert the 1d data into the required image data
    decoded_image = cv2.imdecode(np_1d_buffer, cv2.IMREAD_COLOR)

    if not os.path.exists("requested_images"):
        os.mkdir("requested_images")
    

    # Do the inference
    model_resp = do_the_complete_classification(decoded_image)
    model_resp = model_resp.strip()
    try:
        # Save the image
        cv2.imwrite(f"requested_images/{model_resp}.png",decoded_image)
    except Exception as e:
        print("Error creating image")
        print(e)
    
    #log
    print(f"Req Received and model responsed -> {model_resp}")

    #Resp to user
    st.write(f"#### Model Thinks its a -> <strong>{model_resp}</strong>" , unsafe_allow_html=True)
