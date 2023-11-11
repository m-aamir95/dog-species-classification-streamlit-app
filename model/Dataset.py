import os

import torch
import torchvision.transforms as transforms

import cv2

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class CustomStanfordImageDataset():

  def __init__(self, images_path,device):

    self.device = device
    self.root_dir = images_path

    #Loading all the images along with their labels into memory
    self.images_labels = []

    #Transform which will be applied in order to prepare data for the Neural-Network
    image_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Grayscale() ,
                                           transforms.Resize((64, 64), antialias=None),
                                           transforms.Normalize((0.5), (0.5))])

    #Fitting Ordinal and OneHotEncoder to later encode labels
    self.labelEncoder  = LabelEncoder()
    self.labelEncoder  = self.labelEncoder.fit(os.listdir(self.root_dir))
    label_encoded = self.labelEncoder.transform(os.listdir(self.root_dir))

    self.oneHotEncoder = OneHotEncoder(sparse=False)
    self.oneHotEncoder = self.oneHotEncoder.fit(label_encoded.reshape(len(label_encoded),1))




    # Here _dir will also serve as the label of all the images inside this particular _dir

    for _dir in sorted(os.listdir(self.root_dir)):

      # For Each dir, read all the images, and store them into memory with their corrosponding labels i-e dirname
      for _image in os.listdir(os.path.join(self.root_dir, _dir)):
        image_URI = os.path.join(self.root_dir, _dir, _image)

        try:

          loaded_image = cv2.imread(image_URI)
          transformed_image = image_transforms(loaded_image)

          #Pushing loaded and transformed data into dataset store
          one_hot_encoder_label = torch.tensor(self.oneHotEncoder.transform(self.labelEncoder.transform([_dir]).reshape(1,1)), dtype=torch.float32)
          self.images_labels.append((transformed_image.type(torch.float32), one_hot_encoder_label))

        except exception as e:
          pass

  def __getitem__(self, index):
    image_data, image_label = self.images_labels[index]
    return (image_data.to(self.device), image_label.to(self.device))


  def __len__(self):
    return len(self.images_labels)
