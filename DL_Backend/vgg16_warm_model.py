# Here a warm model means, that we are using vgg16 in its pre-trained state
# We will be only training the head/last-fully-connected-layers to fine-tune the model
# On the stanford dogs species classification dataset


import torchvision
import torch

# The class is just a wrapper around the actual vgg16 model
class PreTrainedVGG16Wrapper():

    # Here the number of classes will modify the output/classification layer of the VGG16
    # Which will be according to the domain specific task
    def __init__(self, num_of_classes : int) -> None:
        
        self.pretrained_VGG16_model = torchvision.models.vgg16_bn(pretrained=True)
        self.num_of_classes = num_of_classes
        
        self.setup_pretrained_network_for_fine_tuning()
    
    def setup_pretrained_network_for_fine_tuning(self):
        
        # Freezing the fully conv layers, these will be not be trained/fine-tuned over the new dataset
        # Freezing conv layers
        for feature in self.pretrained_VGG16_model.features.parameters():
            feature.require_grad = False

        # From the structure of the vgg16 model we know that it contains a classification head
        # We also know that it has six 6 layers and the last layer is the actual classification layer
        # For for more info please print(self.pretrained_VGG16_model)

        # We have not freezed any of the classification head layers
        # However we are going to change the last layer according to our domain specific classification task
        num_features = self.pretrained_VGG16_model.classifier[6].in_features

        # Define your custom classification head
        self.custom_classification_head = torch.nn.Sequential(
            torch.nn.Linear(num_features, 2048),
            torch.nn.Linear(2048, 1024),
            torch.nn.Linear(1024, self.num_of_classes)
        )

        self.pretrained_VGG16_model.classifier[6] = self.custom_classification_head


    def get_warm_vgg16(self):

        return self.pretrained_VGG16_model

    def __str__(self):
        return f"PreTrainedVGG16Wrapper(num_of_classes={self.num_of_classes})"
