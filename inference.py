import gdown
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision

from DL_Backend.model import ConvolutionalNeuralNetwork

import numpy as np

import streamlit as st

from dotenv import load_dotenv


# Class names are not directly available
# we will have to obtain them from dir names
# Of the data
@st.cache_data
def load_classNames() -> [str]:
    base_dir = "./Raw_Data/Images"

    breeds = []

    for d in sorted(os.listdir(base_dir)):

        relative_dir_path = os.path.join(base_dir, d)

        if os.path.isdir(relative_dir_path):

            # Get the parts
            path_parts = d.split('-')
            breed_name = path_parts[1]
            breeds.append(breed_name)
    
    return breeds



# Load model and put it do eval
@st.cache_resource
def load_model():


    model_url = "https://drive.google.com/file/d/183ruW0I5r2GkXGQl1-qO4Hb0oooCc3Au/view?usp=sharing"
    downloaded_model_name = "DL_Backend/dogs_classification_cnn_model.pym"

    if not os.path.exists(downloaded_model_name):
        print("Model not found, downloading....")
        gdown.download(model_url, downloaded_model_name,fuzzy=True)
    else:
        print("Model found!, Skipping download")

    # Instantiate the model
     # Define a dictionary mapping model names to their corresponding classes
    model_classes = {
        'vgg16': torchvision.models.vgg16(pretrained=True),
        'resnet50': torchvision.models.resnet50(pretrained=True),
        'resnet101': torchvision.models.resnet101(pretrained=True),
        "custom_cnn" : ConvolutionalNeuralNetwork(),
        "inception_v3" : torchvision.models.inception_v3(pretrained=True)
        # Add more models as needed
    }

    # Get the respective model for inference depending on the env variables
    model = model_classes[os.getenv("inference_model")].to(torch.device("cpu"))
    model.load_state_dict(torch.load(downloaded_model_name, map_location=torch.device("cpu")))

    # Put the model to eval mode
    model.eval()

    return model

@st.cache_data
def make_prediction(input_image):
    
    # Transform the image to required format
    resize_width = int(os.getenv("resize_width"))
    resize_height = int(os.getenv("resize_height"))

    transforms = T.Compose([T.ToTensor(), 
                            T.Resize((resize_width, resize_height), antialias=None),
                            T.Normalize((0.5), (0.5))])
    transformed_image = transforms(input_image)

    with torch.no_grad():

        #Reshape according to the expected shape of the model Batch x Channels x Width x Height
        transformed_image.reshape(-1, 1, resize_width, resize_height)

        model = load_model()

        prediction = model(transformed_image)

        return prediction



# Will load the actual classnames
# Will do the predictions
# Will convert the model results into human readable classnames
def do_the_complete_classification(image : np.array) -> str:
   
    load_dotenv()
    
    # Load breeds/class names
    breeds = load_classNames()

    predictions = make_prediction(image)

    # Converting to a class from logits
    predictions_softmaxed = F.softmax(predictions, dim=1)
    max_index = torch.argmax(predictions_softmaxed, dim=1)
    return breeds[max_index]
