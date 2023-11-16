# Here a warm model means, that we are using vgg16 in its pre-trained state
# We will be only training the head/last-fully-connected-layers to fine-tune the model
# On the stanford dogs species classification dataset


import torchvision
import torch

# The class is just a wrapper around the actual vgg16 model
class PreTrainedRESNETWrapper():

    # Here the number of classes will modify the output/classification layer of the VGG16
    # Which will be according to the domain specific task
    def __init__(self, num_of_classes : int) -> None:
        
        self.pretrained_resnet_model = torchvision.models.resnet101(pretrained=True)
        self.num_of_classes = num_of_classes
        
        self.setup_pretrained_network_for_fine_tuning()
    
    def setup_pretrained_network_for_fine_tuning(self):
        
        # Freezing the fully conv layers, these will be not be trained/fine-tuned over the new dataset
        # Freezing conv layers
        for param in self.pretrained_resnet_model.parameters():
            param.require_grad = False

        # From the structure of the vgg16 model we know that it contains a classification head
        # We also know that it has six 6 layers and the last layer is the actual classification layer
        # For for more info please print(self.pretrained_resnet_model)

        # We have not freezed any of the classification head layers
        # However we are going to change the last layer according to our domain specific classification task
        num_features = self.pretrained_resnet_model.fc.in_features
        self.pretrained_resnet_model.fc = torch.nn.Linear(num_features, self.num_of_classes)

    def get_warm_resnet(self):


        return self.pretrained_resnet_model

    def __str__(self):
        return f"PreTrainedRESNETWrapper(num_of_classes={self.num_of_classes})"

    