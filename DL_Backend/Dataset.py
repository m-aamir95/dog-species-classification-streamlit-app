import os

import torch
import torchvision.transforms as transforms


import cv2

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class CustomStanfordImageDataset():

  def __init__(self, dogs_breed_dictionary,device):

    self.device = device
    self.dogs_breed_dictionary = dogs_breed_dictionary

    VGG16_REQUIRED_WIDTH = 224
    VGG16_REQUIRED_HEIGHT = 224
    self.resize_width = VGG16_REQUIRED_WIDTH
    self.resize_width = VGG16_REQUIRED_HEIGHT

    #Loading all the images along with their labels into memory
    self.images_and_labels = []

    #Transform which will be applied in order to prepare data for the Neural-Network
    image_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((self.resize_width, self.resize_width), antialias=None),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=2),
                                           transforms.Normalize((0.5), (0.5))])

    #Fitting Ordinal and OneHotEncoder to later encode labels
    self.labelEncoder  = LabelEncoder()
    self.labelEncoder  = self.labelEncoder.fit(list(self.dogs_breed_dictionary.keys()))
    label_encoded = self.labelEncoder.transform(list(self.dogs_breed_dictionary.keys()))

    self.oneHotEncoder = OneHotEncoder(sparse=False)
    self.oneHotEncoder = self.oneHotEncoder.fit(label_encoded.reshape(len(label_encoded),1))



    for dog_breed in sorted(list(self.dogs_breed_dictionary.keys())):

      # For Each dir, read all the images, and store them into memory with their corrosponding labels i-e dirname
      for dog_image_path in self.dogs_breed_dictionary[dog_breed]:
        try:

          loaded_image = cv2.imread(dog_image_path)
          transformed_image = image_transforms(loaded_image)

          #Pushing loaded and transformed data into dataset store
          one_hot_encoder_label = torch.tensor(self.oneHotEncoder.transform(self.labelEncoder.transform([dog_breed]).reshape(1,1)), dtype=torch.float32)
          self.images_and_labels.append((transformed_image.type(torch.float32), one_hot_encoder_label))

        except:
          print("Exeception while processing and loading image from disk")



  def __getitem__(self, index):
    image_data, image_label = self.images_and_labels[index]
    return (image_data.to(self.device), image_label.to(self.device))


  def __len__(self):
    return len(self.images_and_labels)
