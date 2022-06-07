from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torchvision.transforms.functional as TF
from PIL import Image

class Mango(Dataset):
	def __init__(self, data='train', transform=None):
		root = '/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/'
		csv_file = ''
		self.image_root = ''
		if data == 'train':
			csv_file = root + 'train.csv'
			self.image_root = root + 'Masked_Train/mask'
		elif data == 'dev':
			csv_file = root + 'dev.csv'
			self.image_root = root + 'Masked_Dev/mask'
		else:
			print('fail to load data')

		self.data = pd.read_csv(csv_file)
		self.transform = transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = self.image_root + self.data.image_id[idx]
		image = io.imread(img_name)
		image = Image.fromarray(image)
		if self.transform:
			image = self.transform(image)

		label = self.data.label[idx]
		label_onehot = []
		if label == 'A':
			label_onehot = 0
		elif label == 'B':
			label_onehot = 1
		else:
			label_onehot = 2

		return image, label_onehot,img_name

class Rotate(object):
	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TF.rotate(x, self.angle)

def show_batch(sample_batched, title):
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(title)
    plt.axis('off')


if __name__ == '__main__':
	batch_size = 16
	transform = transforms.Compose([
		transforms.Resize([256, 256]),
		Rotate(15),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
	dataset = Mango(data='dev', transform=transform)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
	iterator = iter(dataloader)
	images, labels = next(iterator)
	print(labels)
	# plot1 = plt.figure()
	# show_batch(images, 'data')
	# plt.show()