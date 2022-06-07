import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from MangoDataset import *
import matplotlib.pyplot as plt
from model import *
import os
import sys

if __name__ == '__main__':
	name = 'squeezenet'
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#print('device: ', device)

	model =initialize_model("squeezenet",3,True,use_pretrained = False)
	model.to(device)
	model.load_state_dict(torch.load("./model_dict/squeezenet_lr0.005_bsize64_epoch100_unfreeze-total_momentum0.9_sharp + blur data + input_size=224_removecrop_.pth"))
	#####
	#model.classifier[0] = nn.Identity()
	model.classifier[1] = nn.Identity() ###squeezenet
	#model.classifier[6] = nn.Identity()
	#model.classifier = nn.Identity()
	#model.fc = nn.Identity()
	#print(model)


	batch_size = 64
	transform = transforms.Compose([
		transforms.Resize([224, 224]),
		
		transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
	#dataset_dev = get_mango_dataset(data='dev', mask=True, transform='default')
	#dataset_test = get_mango_dataset(data='test', mask=True, transform='default')
	dataset_train = get_mango_dataset(data='train', mask=True, transform='default')
	dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=6)

	cat = torch.tensor([]).to(device)

	with torch.no_grad():
		for batch_idx, (images,labels) in enumerate(dataloader):###labels
			images = images.to(device)			
			features = model(images)
			print(features.shape)
			cat = torch.cat((cat, features), 0)
		cat = cat.to(torch.device("cpu"))
		torch.save(cat, name + 'train_0612.pt')