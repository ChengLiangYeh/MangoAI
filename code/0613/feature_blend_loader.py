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

        # self.name_list = name_list
        data_root = "/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed"

        if data == 'train':
            csv_pth = os.path.join(data_root, 'train.csv')
            self.image_root = os.path.join(data_root,'Masked_Train')
            self.f =  torch.from_numpy(torch.load('/Users/jimmy/Desktop/machine_learning/tech/final/PCA_train.pt'))
        else:
            csv_pth = os.path.join(data_root, 'dev.csv')
            self.image_root = os.path.join(data_root, 'Masked_Dev')
            self.f = torch.from_numpy(torch.load('/Users/jimmy/Desktop/machine_learning/tech/final/PCA_dev.pt'))

        self.data = pd.read_csv(csv_pth)
        self.label_dict = {'A':0, 'B':1, 'C':2}

      
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_pth = os.path.join(self.image_root , "mask"+self.data.image_id[idx])
        #print(img_pth)
        img = Image.open(img_pth).convert("RGB")

        if self.transform:
            img = self.transform(img)
         # no transform for tensor!


    
        label = self.data.label[idx]
        label = self.label_dict[label]

        return img, label, self.f[idx]

# class Rotate(object):
#   def __init__(self, angle):
#       self.angle = angle

#   def __call__(self, x):
#       return TF.rotate(x, self.angle)

def show_batch(sample_batched, title):
    grid = utils.make_grid(sample_batched)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(title)
    plt.axis('off')

def get_mango_dataset(data='train', mask=True, transform='default'):
    

    dataset = Mango(data=data, mask=mask, transform=data_transform)

    return dataset


if __name__ == '__main__':
    batch_size = 40
    name_list =["all .pt/alexnetdev_0611.pt","all .pt/densenet121dev_0611.pt"]
    dataset = Mango(name_list=name_list)
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