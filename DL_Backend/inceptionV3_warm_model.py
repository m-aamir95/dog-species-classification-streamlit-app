# Here a warm model means, that we are using vgg16 in its pre-trained state
# We will be only training the head/last-fully-connected-layers to fine-tune the model
# On the stanford dogs species classification dataset


import torchvision
import torch
import torch.nn as nn

# The class is just a wrapper around the actual model
class PreTrainedInceptionV3Wrapper():

    # Here the number of classes will modify the output/classification layer of the model
    # Which will be according to the domain specific task
    def __init__(self, num_of_classes : int) -> None:
        
        self.pretrained_inception_v3_model = torchvision.models.inception_v3(pretrained=True)
        self.num_of_classes = num_of_classes
        
        self.setup_pretrained_network_for_fine_tuning()
    
    def setup_pretrained_network_for_fine_tuning(self):
        
        # Freezing the fully conv layers, these will be not be trained/fine-tuned over the new dataset
        # Freezing conv layers
        for feature in self.pretrained_inception_v3_model.parameters():
            feature.require_grad = False

        # From the structure of the model as we know that it contains a classification head
        # We are going to modify it
        num_features = self.pretrained_inception_v3_model.fc.in_features

         # Define your custom classification head 1
        self.custom_classification_head_1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),  # Adding ReLU activation function
            nn.Dropout(0.5),  # Adding Dropout with a dropout probability of 0.5
            nn.Linear(1024, 512),
            nn.ReLU(),  # Adding ReLU activation function
            nn.Dropout(0.5),  # Adding Dropout with a dropout probability of 0.5
            nn.Linear(512, self.num_of_classes)
        )

        for f in self.custom_classification_head_1.parameters():
            f.requires_grad = True


        self.pretrained_inception_v3_model.fc = self.custom_classification_head_1

        num_features = self.pretrained_inception_v3_model.AuxLogits.fc.in_features

        # Define your custom classification head 2
        self.custom_classification_head_2 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),  # Adding ReLU activation function
            nn.Dropout(0.5),  # Adding Dropout with a dropout probability of 0.5
            nn.Linear(1024, 512),
            nn.ReLU(),  # Adding ReLU activation function
            nn.Dropout(0.5),  # Adding Dropout with a dropout probability of 0.5
            nn.Linear(512, self.num_of_classes)
        )


        for f in self.custom_classification_head_2.parameters():
            f.requires_grad = True

        self.pretrained_inception_v3_model.AuxLogits.fc = self.custom_classification_head_2


    def get_warm_inception_v3(self):

        return self.pretrained_inception_v3_model

    def __str__(self):
        return f"PreTrainedInception_V3_Wrapper(num_of_classes={self.num_of_classes})"
