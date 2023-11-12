import gdown
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T


from DL_Backend.Dataset import CustomStanfordImageDataset
from DL_Backend.model import ConvolutionalNeuralNetwork

import cv2

# Class names are not directly available
# we will have to obtain them from dir names
# Of the data
def load_classNames() -> [str]:
    base_dir = "./Raw_Data"

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
def load_model() -> ConvolutionalNeuralNetwork:

    model_url = "https://drive.google.com/file/d/1qaNErdLEUslRutwl0_H9-bBjPc-gHTK-/view"
    downloaded_model_name = "DL_Backend/dogs_classification_cnn_model.pym"

    if not os.path.exists(downloaded_model_name):
        gdown.download(model_url, downloaded_model_name,fuzzy=True)
    else:
        print("Model found!, Skipping download")

    # Instantiate the model
    model = ConvolutionalNeuralNetwork().to(torch.device("cpu"))
    model.load_state_dict(torch.load(downloaded_model_name, map_location=torch.device("cpu")))

    # Put the model to eval mode
    model.eval()

    return model

def make_prediction(input_image):

    # Transform the image to required format
    resize_width = 64
    resize_height = 64

    transforms = T.Compose([T.ToTensor(), 
                            T.Grayscale(), 
                            T.Resize((resize_width, resize_height), antialias=None),
                            T.Normalize((0.5), (0.5))])
    transformed_image = transforms(input_image)

    with torch.no_grad():

        #Reshape according to the expected shape of the model Batch x Channels x Width x Height
        transformed_image.reshape(-1, 1, resize_width, resize_height)

        model = load_model()

        prediction = model(transformed_image)

        return prediction



if __name__ == "__main__":

    # Load breeds/class names
    breeds = load_classNames()

    image_path = "Raw_Data/n02085620-Chihuahua/n02085620_199.jpg"
    dog_image = cv2.imread(image_path)
    
    predictions = make_prediction(dog_image)

    # Converting to a class from logits
    predictions_softmaxed = F.softmax(predictions, dim=1)
    max_index = torch.argmax(predictions_softmaxed, dim=1)
    print(max_index)
    print(breeds)
    print(breeds[max_index])
