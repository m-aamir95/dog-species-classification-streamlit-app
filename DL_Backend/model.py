import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):

  def __init__(self):
    super().__init__()

    # Convolutional layers
                                            # Input Image  -> 128 * 128 * 1
    self.conv1 = nn.Conv2d(1,   201, (3,3)) # Output Image -> 126 * 126 * 201
    self.conv2 = nn.Conv2d(201, 12,  (3,3)) # Output Image -> 124 * 124 * 12
    self.conv3 = nn.Conv2d(12,  120, (3,3)) # Output Image -> 122 * 122 * 120
    self.conv4 = nn.Conv2d(120, 209, (3,3)) # Output Image -> 120 * 120 * 209

    print(f" Last CONV4 object => {self.conv4}")

    # In order to know the output shape of last conv layer, we would be generating some dummy data, passing through all the conv layers
    # And will the store the output of the final conv layer
    x = torch.randn((64,64)).view(-1, 1, 64, 64)
    self.to_linear = None
    self.convs(x)

    print(f" Final output of convolutional-layer (My-Calculation)=> {self.to_linear}")

    # Fully connected layers
    self.fc1   = nn.Linear(self.to_linear, 201)
    self.fc2   = nn.Linear(201, 209)
    self.fc3   = nn.Linear(209, 120)

  def convs(self, X):
    #Pass through all the convolution layers
    X = F.relu(self.conv1(X))
    X = F.relu(self.conv2(X))
    X = F.relu(self.conv3(X))
    X = F.relu(self.conv4(X))

    if self.to_linear is None:
      self.to_linear = X[0].shape[0] * X[0].shape[1] * X[0].shape[2]
    return X


  def forward(self, X):

    X = self.convs(X)
    # Flatten the Conv2 layer
    X = X.view(-1, self.to_linear)

    #Pass through all the Linear layers
    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)

    return X


