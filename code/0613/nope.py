import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
# import torchvision.transforms.functional as TF
from PIL import Image
from itertools import cycle, islice

class Mango(Dataset):
	def __init__(self, root='MangoData', data='train', mask=True, transform=None):
		data_root = root
		if mask:
			#self.maskroot = 'clache'#
			self.maskroot = 'mask'
		else:
			self.maskroot = ''

		if data == 'train':
			csv_pth = os.path.join(data_root, 'train.csv')
			self.image_root = os.path.join(data_root, self.maskroot + 'Train')
		else:
			csv_pth = os.path.join(data_root, 'dev.csv')
			self.image_root = os.path.join(data_root, self.maskroot + 'Dev')

		self.data = pd.read_csv(csv_pth)
		self.transform = transform
		self.label_dict = {'A':0, 'B':1, 'C':2}

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_pth = os.path.join(self.image_root, self.maskroot + self.data.image_id[idx])
		#print(img_pth)
		img = Image.open(img_pth).convert("RGB")

		if self.transform:
			img = self.transform(img)

		label = self.data.label[idx]
		label = self.label_dict[label]

		return img, label

# class Rotate(object):
# 	def __init__(self, angle):
# 		self.angle = angle

# 	def __call__(self, x):
# 		return TF.rotate(x, self.angle)

def show_batch(sample_batched, title):
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(title)
    plt.axis('off')

def get_mango_dataset(data='train', mask=True, transform='default'):
	if transform != 'default':
		data_transform = transform
	else:
		if data == 'train':

			transform = [transforms.RandomResizedCrop(224), transforms.RandomRotation(15), transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]
			data_transform = transforms.Compose([
				transforms.RandomOrder(transform),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
			
			'''
			data_transform = transforms.Compose([
				transforms.RandomResizedCrop(224),  #inception = 299 else=224  感覺原本224太小了！改512
				transforms.RandomRotation(15),
				transforms.RandomHorizontalFlip(),
				transforms.RandomVerticalFlip(),
				#transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0),   用了變差！
				#transforms.Grayscale(num_output_channels=3),   #####記得改!
				#transforms.RandomGrayscale(p=0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
			'''

		else:
			data_transform = transforms.Compose([
				transforms.Resize([224, 224]), #inception = 299 else=224
				#transforms.Grayscale(num_output_channels=3),
				#transforms.RandomGrayscale(p=0.5),
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])

	dataset = Mango(data=data, mask=mask, transform=data_transform)

	return dataset


if __name__ == '__main__':
	batch_size = 40
	dataset = get_mango_dataset(data='Dev', mask=True, transform='default')
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
	iterator = cycle(dataloader)
	print(type(iterator))
	for idx, (_, labels) in enumerate(islice(iterator, 80)):
		print(labels)
	# images, labels = next(iterator)
	# print(images.shape)
	# print(labels)
	# plot1 = plt.figure()
	# show_batch(images, 'data')
	# plt.show()