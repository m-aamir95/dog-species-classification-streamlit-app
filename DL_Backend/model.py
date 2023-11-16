import torch
import torch.nn as nn
import torch.nn.functional as F

#region My Custom Very Small CNN, does not produce good Testing accuracies
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 201, (3, 3))
        self.conv2 = nn.Conv2d(201, 12, (3, 3))
        self.conv3 = nn.Conv2d(12, 120, (3, 3))
        self.conv4 = nn.Conv2d(120, 209, (3, 3))

        # In order to know the output shape of the last conv layer
        x = torch.randn((64, 64)).view(-1, 1, 64, 64)
        self.to_linear = None
        self.convs(x)

        # Fully connected layers
        self.fc1 = nn.Linear(self.to_linear, 201)
        self.fc2 = nn.Linear(201, 209)
        self.fc3 = nn.Linear(209, 120)
        self.dropout = nn.Dropout(0.5)

    def convs(self, X):
        # Pass through all the convolution layers
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = F.relu(self.conv4(X))

        if self.to_linear is None:
            self.to_linear = X[0].shape[0] * X[0].shape[1] * X[0].shape[2]
        return X

    def forward(self, X):
        X = self.convs(X)
        X = X.view(-1, self.to_linear)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.dropout(X)
        return X

    def __str__(self):
        return "CustomCNNModel"

#endregion

