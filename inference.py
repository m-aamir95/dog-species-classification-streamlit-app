import gdown
import torch
import os

from model.Dataset import CustomStanfordImageDataset
from model.model import ConvolutionalNeuralNetwork

def main():

    model_url = "https://drive.google.com/file/d/1qaNErdLEUslRutwl0_H9-bBjPc-gHTK-/view"
    downloaded_model_name = "dogs_classification_cnn_model.pym"

    if not os.path.exists(downloaded_model_name):
        gdown.download(model_url, downloaded_model_name,fuzzy=True)
    else:
        print("Model found!, Skipping download")

    # Instantiate the model
    model = ConvolutionalNeuralNetwork().to(torch.device("cpu"))
    model.load_state_dict(torch.load(downloaded_model_name, map_location=torch.device("cpu")))


    

if __name__ == "__main__":
    main()