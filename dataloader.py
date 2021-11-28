import os
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

class Image_Loader(Dataset):
    def __init__(self, root_path='./train_set.csv', image_size=[32, 32], transforms_data=True):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # read the images from image path
        image_path = os.path.join(self.data_path.iloc[idx, 0])
        image = Image.open(image_path)
        width, height = image.size
        image = image.crop((0, height/2, width, height))
        
        # read label
        label_cross = self.data_path.iloc[idx, 1]

        if self.transforms_data == True:
            data_transform = self.transform(False, True, True)
            image = data_transform(image)

        return image, torch.from_numpy(np.array(label_cross, dtype=np.long))

    def transform(self, flip, resize, totensor):
        options = []
        if flip:
            options.append(transforms.RadomHorizontalFlip())
        if resize:
            options.append(transforms.Resize(self.image_size))
        if totensor:
            options.append(transforms.ToTensor())
        
        transform = transforms.Compose(options)

        return transform
class Image_Loader_test(Dataset):
    def __init__(self, root_path='./test_set.csv', image_size=[32,32], transforms_data=True):
        
        self.data_path = pd.read_csv(root_path)
        self.image_size = image_size
        self.num_images = len(self.data_path)
        self.transforms_data = transforms_data
        
    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        # read the images from image path
        image_path = os.path.join(self.data_path.iloc[idx, 0])
        image = Image.open(image_path)
        
        # read label
        label_cross = self.data_path.iloc[idx, 1]

        if self.transforms_data == True:
            data_transform = self.transform(False, True, True)
            image = data_transform(image)

        return image, torch.from_numpy(np.array(label_cross, dtype=np.long))

    def transform(self, flip, resize, totensor):
        options = []
        if flip:
            options.append(transforms.RandomHorizontalFlip())
        if resize:
            options.append(transforms.Resize(self.image_size))
        if totensor:
            options.append(transforms.ToTensor())
        
        transform = transforms.Compose(options)

        return transform

    
