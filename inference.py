import gdown
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision

from DL_Backend.model import ConvolutionalNeuralNetwork
from DL_Backend.inceptionV3_warm_model import PreTrainedInceptionV3Wrapper

import numpy as np

import streamlit as st

from dotenv import load_dotenv


# Class names are not directly available
# we will have to obtain them from dir names
# Of the data
# Loading dog breed names/classes
@st.cache_data
def load_classNames() -> [str]:
    dog_brred_filename = "dog_breed_names.txt"

    # breeds = []

    # for d in sorted(os.listdir(base_dir)):

    #     relative_dir_path = os.path.join(base_dir, d)

    #     if os.path.isdir(relative_dir_path):
    #         # Get the parts
    #         path_parts = d.split('-')
    #         breed_name = path_parts[1]
    #         breeds.append(breed_name)

    breeds = None
    with open(dog_brred_filename, "r") as file:
        breeds = file.readlines()
    
    return breeds



# Load model and put it do eval
@st.cache_resource
def load_model():


    model_url = "https://drive.google.com/file/d/1Tazkj_ZcCHLAszxF4APy_jgLsSefZkvi/view?usp=sharing"
    downloaded_model_name = "DL_Backend/dogs_classification_cnn_model.pym"

    if not os.path.exists(downloaded_model_name):
        print("Model not found, downloading....")
        gdown.download(model_url, downloaded_model_name,fuzzy=True)
    else:
        print("Model found!, Skipping download")

    # Get the respective model for inference depending on the env variables

    model = PreTrainedInceptionV3Wrapper(num_of_classes=120, load_pretrained_warm_model=False).get_warm_inception_v3().to(torch.device("cpu"))
    model.load_state_dict(torch.load(downloaded_model_name, map_location=torch.device("cpu")))

    # Put the model to eval mode
    model.eval()

    return model

@st.cache_data
def make_prediction(input_image):
    
    # Transform the image to required format
    resize_width = int(os.getenv("resize_width"))
    resize_height = int(os.getenv("resize_height"))

    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Resize((resize_width, resize_height), antialias=None),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transformed_image = transforms(input_image)

    with torch.no_grad():

        #Reshape according to the expected shape of the model Batch x Channels x Width x Height
        transformed_image = transformed_image.reshape(1,3, resize_width, resize_height)

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

