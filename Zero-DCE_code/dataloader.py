import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
import cv2

random.seed(1143)


def populate_train_list(lowlight_images_path):
    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.TIFF']

    image_list_lowlight = []
    for ext in supported_extensions:
        image_list_lowlight.extend(glob.glob(os.path.join(lowlight_images_path, ext)))

    random.shuffle(image_list_lowlight)

    return image_list_lowlight

class lowlight_loader(data.Dataset):

	def __init__(self, lowlight_images_path):

		self.train_list = populate_train_list(lowlight_images_path)
		self.size = 256

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))




	def __getitem__(self, index):

		data_lowlight_path = self.data_list[index]

		data_lowlight = Image.open(data_lowlight_path)

		# data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
		data_lowlight = data_lowlight.resize((self.size, self.size), Image.Resampling.LANCZOS)

		data_lowlight = (np.asarray(data_lowlight)/255.0)
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

