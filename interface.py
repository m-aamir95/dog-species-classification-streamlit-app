"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd

# Set page configuration to wide mode
st.set_page_config(layout="wide")

st.title("Dog Species Classification Via Convolutional Neural Networks")
# Section 1: Introduction
st.header("How it works")
st.write("##### The app is leveraging a custom built CNN model for classifying dog species. "
         "The model has been trained on the [Stanford dog breeds dataset](http://vision.stanford.edu/aditya86/ImageNetDogs). To achieve the best results, use images where the dog is easily distinguishable.")

# Section 2: Samples Dog images
st.header("Samples Images")



# Section 3: Conclusion
st.header("Conclusion")
st.write("##### Summarize your findings and conclude your Streamlit app.")

