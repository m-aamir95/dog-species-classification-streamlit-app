# Instead of using torchvision.datasets.ImageFolder
# I will be writing my custom-dataset to load data into memory and convert into pytorch compatible format
from DL_Backend.Dataset import CustomStanfordImageDataset
from DL_Backend.model import ConvolutionalNeuralNetwork

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


# Hyper parameters
images_path = "/content/Raw_Data/Images"

lr = 0.02
batch_size = 256 
epocs = 50
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():

    dataset = CustomStanfordImageDataset(images_path=images_path,device=device)
    print("Custom Dataset Initialized")
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader Initialized")

    model =  ConvolutionalNeuralNetwork().to(device)
    print("Model Initialized")



    # Instantiating optimizer and passig lr and network parameters to fine-tune
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i in range(epocs):

        correct_count = 0
        wrong_count = 0
        accuracies_count = 0
        overall_epoc_accuracy = [] # Mean of all accuracies for the current epoc
        for (X_features_batch, Y_labels_batch) in data_loader:

            train_x, train_y = X_features_batch, Y_labels_batch

            model.zero_grad()

            predictions = model(train_x)

            # Reshapping prediction(Y_hat) to match Y
            train_y = train_y.view(predictions.shape)
            _decoded_train_y = dataset.oneHotEncoder.inverse_transform(train_y.to("cpu").detach().numpy())
            loss = F.cross_entropy(predictions, torch.tensor(_decoded_train_y, dtype=torch.long).view(len(_decoded_train_y)).to(device))

            #Calculate derivative
            loss.backward()

            #Update weights
            optimizer.step()



        print(f"Itr # {i}, Loss => {loss.item()}")

        torch.save(model.state_dict(), "dog_species_classification_model.pym")




if __name__ == "__main__":

    main()