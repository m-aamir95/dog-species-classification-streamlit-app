import os

# Instead of using torchvision.datasets.ImageFolder
# I will be writing my custom-dataset to load data into memory and convert into pytorch compatible format
from DL_Backend.Dataset import CustomStanfordImageDataset
from DL_Backend.model import ConvolutionalNeuralNetwork
from DL_Backend.vgg16_warm_model import PreTrainedVGG16Wrapper
from DL_Backend.resnet_warm_model import PreTrainedRESNETWrapper
from DL_Backend.data_preprocess import custom_train_test_split

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

import wandb

from dotenv import load_dotenv


def main():

    load_dotenv()

    images_path = "Raw_Data/Images"

    # Hyper parameters
    lr = float(os.getenv("LR"))
    batch_size = int(os.getenv("BATCHES"))
    epocs = int(os.getenv("EPOCHS"))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    resize_width = int(os.getenv("resize_width"))
    resize_height = int(os.getenv("resize_height"))

    # Load the custom splitted dataset
    # This will only load the image paths
    # The actual images will be loaded and further processed into the Pytorch Dataset
    train_test_dataset = custom_train_test_split(data_root_dir=images_path, train_size=0.9)

    # Init the training dataset and dataloader
    train_image_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((resize_width, resize_height), antialias=None),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=5),
                                           transforms.Normalize((0.5), (0.5))])
    
    train_dataset = CustomStanfordImageDataset(train_test_dataset["train"],transforms=train_image_transforms,device=device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Init the testing dataset and dataloader

    test_image_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((resize_width, resize_height), antialias=None),
                                        transforms.Normalize((0.5), (0.5))])
    test_dataset = CustomStanfordImageDataset(train_test_dataset["test"],transforms=test_image_transforms,device=device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    
    # model =  ConvolutionalNeuralNetwork().to(device)
    model_wrapper = PreTrainedVGG16Wrapper(num_of_classes=120)
    model = model_wrapper.get_warm_vgg16().to(device)


    # Instantiating optimizer and passig lr and network parameters to fine-tune
    optimizer = optim.SGD(model.parameters(), lr=lr)

     # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="Dog Species Classification Streamlit App",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": lr,
        "epochs": epocs,
        "architecture": "CNN-based",
        "model" : str(model_wrapper),
        "optimizer" : str(optimizer),
        "train_size" : str(len(train_dataset)),
        "test_size" : str(len(test_dataset)),
        "batch_size" : batch_size,
        "dataset": "Stanford Dog Species",
        "device" : str(device)
        }
    )

    for i in range(epocs):

        # Put the model to train mode
        model.train()
     
         # Get the Results on the train set
        total_train_correct_samples = 0
        total_train_samples = 0
        total_train_loss = 0

       
        for (X_features_batch, Y_labels_batch) in train_dataloader:

            train_x, train_y = X_features_batch, Y_labels_batch

            model.zero_grad()

            y_hat_train = model(train_x)

            print(y_hat_train.shape)
            print(train_y.shape)
            # Reshapping prediction(Y_hat) to match Y
            # Is required to compute the loss
            train_y = train_y.view(y_hat_train.shape)

            # We are converting the one hot vector back to the label e.g, 1,2,3
            # Because the CrossEntropy Loss function In Pytorch expects Y to be labels
            # And Y_HAT to be one hot vectors
            # While we are at it, I dont think we need to convert the labels to one hot vectors
            # Ourselves because for Y we need labels not one hot vectors
            # For more info please watch the following awesome video at 7 min onwards
            # to get a better idea

            # https://www.youtube.com/watch?v=7q7E91pHoW4&ab_channel=PatrickLoeber

            # TODO; Potential bottleneck moving data from CPU
            _decoded_train_y = train_dataset.oneHotEncoder.inverse_transform(train_y.to("cpu").detach().numpy())
            train_loss = F.cross_entropy(y_hat_train, torch.tensor(_decoded_train_y, dtype=torch.long).view(len(_decoded_train_y)).to(device))

            total_train_loss += train_loss.item()

            # Applying softmax to y_hat because we are about to compare with the original y
            # Softmax and then rouding will help to better compare with original vector

            y_hat_train_rounded_softmaxed = torch.round(F.softmax(y_hat_train, dim=1))

            #Compare how many one-hot vectors or predictions match
            matching_elems = torch.eq(train_y, y_hat_train_rounded_softmaxed) # Will produce a matrix of true and 
                                                                             # false value where value match
                                                                             # or not match
                                
            # torch.all() will make sure for matching
            # data on a given axis
            matching_rows = torch.sum(torch.all(matching_elems, dim=1)).item()

            total_train_samples += train_y.shape[0]

            total_train_correct_samples += matching_rows

            #Calculate derivative
            train_loss.backward()

            #Update weights
            optimizer.step()

    
        print(f"Itr # {i}, Loss => {train_loss.item()}")

        torch.save(model.state_dict(), "dog_species_classification_model.pym")

        # Get the Results on the test set
        total_test_correct_samples = 0
        total_test_samples = 0
        total_test_loss = 0

        #Put the model to eval mode
        model.eval()

        with torch.no_grad():

            for (X_test_features_batch, Y_test_labels_batch) in test_dataloader:
            
                total_test_samples += Y_test_labels_batch.shape[0]

                x_test , y_test = X_test_features_batch, Y_test_labels_batch

                y_hat_test = model(x_test)

                # print(y_hat_test.shape)
                # print(y_test.shape)

                # Reshapping prediction(Y_hat) to match Y
                # Is required to compute the loss
                y_test = y_test.view(y_hat_test.shape)

        


                # We are converting the one hot vector back to the label e.g, 1,2,3
                # Because the CrossEntropy Loss function In Pytorch expects Y to be labels
                # And Y_HAT to be one hot vectors
                # While we are at it, I dont think we need to convert the labels to one hot vectors
                # Ourselves because for Y we need labels not one hot vectors
                # For more info please watch the following awesome video at 7 min onwards
                # to get a better idea

                # https://www.youtube.com/watch?v=7q7E91pHoW4&ab_channel=PatrickLoeber

                # TODO; Potential bottleneck moving data from CPU
                _decoded_test_y = test_dataset.oneHotEncoder.inverse_transform(y_test.to("cpu").detach().numpy())
                test_loss = F.cross_entropy(y_hat_test, torch.tensor(_decoded_test_y, dtype=torch.long).view(len(_decoded_test_y)).to(device))

                total_test_loss += test_loss.item()

                # Applying softmax to y_hat because we are about to compare with the original y
                # Softmax and then rouding will help to better compare with original vector

                y_hat_test_rounded_softmaxed = torch.round(F.softmax(y_hat_test, dim=1))

                #Compare how many one-hot vectors or predictions match
                matching_elems = torch.eq(y_test, y_hat_test_rounded_softmaxed) # Will produce a matrix of true and 
                                            # false value where value match
                                            # or not match
                                
                                # torch.all() will make sure for matching
                                # data on a given axis
                matching_rows = torch.sum(torch.all(matching_elems, dim=1)).item()

                total_test_samples += y_test.shape[0]

                total_test_correct_samples += matching_rows

                # break

            print(f"Total test loss -> {total_test_loss}")
            print(f"Total test Accuracy -> {total_test_correct_samples/total_test_samples}") 

            # log metrics to wandb
            wandb.log({"train_acc": (total_train_correct_samples/total_train_samples), 
                       "train_loss" : total_train_loss,
                       "test_acc": (total_test_correct_samples/total_test_samples),
                         "test_loss": total_test_loss}) 
    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == "__main__":

    main()
