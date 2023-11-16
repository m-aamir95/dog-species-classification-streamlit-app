import os
import math

def custom_train_test_split(data_root_dir : str, train_size : int = 0.8) -> dict:
   
    train_size = train_size
    test_size = 1.0 - train_size

    # I am going to create a dictionary like below
    # {
    #     "train" : {"breed_A":[], "breed_B":[] ,"breed_C":[], "breed_D":[]}

    #     "test" :  {"breed_A":[], "breed_B":[] ,"breed_C":[], "breed_D":[]}
    # }

    # This structure.train or structure.test will be passed to dataset to create the Pytorch Dataset
   
    structure = {"train": {}, "test":{}}

    for sub_dir in sorted(os.listdir(data_root_dir)):

        dog_breed_dir = os.path.join(data_root_dir, sub_dir)
        
        # Insert the dog_breed into the structure for both train and test set
        structure["train"][sub_dir] = []
        structure["test"][sub_dir] = []

        num_of_train_images_to_insert = math.floor(len(os.listdir(dog_breed_dir)) * train_size)
        num_of_test_images_to_insert = len(os.listdir(dog_breed_dir)) - num_of_train_images_to_insert
        
        inserted_train_images = 0
        for img in sorted(os.listdir(dog_breed_dir)):

            img_full_path = os.path.join(dog_breed_dir, img)
            
            # TODO; I believe I should introduce some randomness regarding how the data is pushed in train and test sets
            # TODO; right now first train% is always pushed inside the train set
            # TODO; and the rest is pushed into the test
            if inserted_train_images < num_of_train_images_to_insert:
                structure["train"][sub_dir].append(img_full_path)
            else:
                structure["test"][sub_dir].append(img_full_path)
            
            inserted_train_images+=1

    return structure
