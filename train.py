# Instead of using torchvision.datasets.ImageFolder
# I will be writing my custom-dataset to load data into memory and convert into pytorch compatible format
from DL_Backend.Dataset import CustomStanfordImageDataset
from DL_Backend.model import ConvolutionalNeuralNetwork
from DL_Backend.data_preprocess import custom_train_test_split

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


# Hyper parameters
images_path = "Raw_Data/Images"

lr = 0.02
batch_size = 256 
epocs = 50
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def main():

    # Load the custom splitted dataset
    # This will only load the image paths
    # The actual images will be loaded and further processed into the Pytorch Dataset
    train_test_dataset = custom_train_test_split(data_root_dir=images_path, train_size=0.8)

    # Init the training dataset and dataloader
    train_dataset = CustomStanfordImageDataset(train_test_dataset["train"],device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Init the testing dataset and dataloader
    test_dataset = CustomStanfordImageDataset(train_test_dataset["test"],device=device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    model =  ConvolutionalNeuralNetwork().to(device)


    # Instantiating optimizer and passig lr and network parameters to fine-tune
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i in range(epocs):

        # Put the model to train mode
        model.train()

        correct_count = 0
        wrong_count = 0
        accuracies_count = 0
        overall_epoc_accuracy = [] # Mean of all accuracies for the current epoc
        for (X_features_batch, Y_labels_batch) in train_dataloader:

            train_x, train_y = X_features_batch, Y_labels_batch

            model.zero_grad()

            predictions = model(train_x)

            # Reshapping prediction(Y_hat) to match Y
            # Is required to compute the loss
            train_y = train_y.view(predictions.shape)

            # We are converting the one hot vector back to the label e.g, 1,2,3
            # Because the CrossEntropy Loss function In Pytorch expects Y to be labels
            # And Y_HAT to be one hot vectors
            # While we are at it, I dont think we need to convert the labels to one hot vectors
            # Ourselves because for Y we need labels not one hot vectors
            # For more info please watch the following awesome video at 7 min onwards
            # to get a better idea

            # https://www.youtube.com/watch?v=7q7E91pHoW4&ab_channel=PatrickLoeber
            _decoded_train_y = train_dataset.oneHotEncoder.inverse_transform(train_y.to("cpu").detach().numpy())
            loss = F.cross_entropy(predictions, torch.tensor(_decoded_train_y, dtype=torch.long).view(len(_decoded_train_y)).to(device))

            #Calculate derivative
            loss.backward()

            #Update weights
            optimizer.step()

            break # TODO remove it


        print(f"Itr # {i}, Loss => {loss.item()}")

        torch.save(model.state_dict(), "dog_species_classification_model.pym")

        # Get the Results on the test set
        total_test_accuracy = 0
        total_test_samples = 0
        total_test_loss = 0

        #Put the model to eval mode
        model.eval()

        with torch.no_grad():

            for (X_test_features_batch, Y_test_labels_batch) in test_dataloader:
            
                total_test_samples += Y_labels_batch.shape[0] # TODO verify the shape

                x_test , y_test = X_test_features_batch, Y_test_labels_batch

                y_hat_test = model(x_test)

                # We are converting the one hot vector back to the label e.g, 1,2,3
                # Because the CrossEntropy Loss function In Pytorch expects Y to be labels
                # And Y_HAT to be one hot vectors
                # While we are at it, I dont think we need to convert the labels to one hot vectors
                # Ourselves because for Y we need labels not one hot vectors
                # For more info please watch the following awesome video at 7 min onwards
                # to get a better idea

                # https://www.youtube.com/watch?v=7q7E91pHoW4&ab_channel=PatrickLoeber
                _decoded_test_y = test_dataset.oneHotEncoder.inverse_transform(y_test.to("cpu").detach().numpy())
                loss = F.cross_entropy(y_hat_test, torch.tensor(_decoded_test_y, dtype=torch.long).view(len(_decoded_test_y)).to(device))

                total_test_loss += loss

                # Applying softmax to y_hat because we are about to compare with the original y
                # Softmax and then rouding will help to better compare with original vector
                F.softmax(y_hat_test,)
        




if __name__ == "__main__":

    main()
