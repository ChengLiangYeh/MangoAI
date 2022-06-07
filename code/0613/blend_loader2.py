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
    def __init__(self,name_list):

        self.name_list = name_list
        csv_pth = "/Users/jimmy/Desktop/machine_learning/tech/final/C1-P1_Train Dev_fixed/dev.csv"
    
        self.data = pd.read_csv(csv_pth)
        self.label_dict = {'A':0, 'B':1, 'C':2}

        self.x = torch.load(self.name_list[0])
        blend = torch.zeros(len(self.x),3)
        for i,name in enumerate(self.name_list):
            if i == 0:
                 self.x = torch.load(name)
            else:
                features_train = torch.load(name)
                self.x = torch.cat((self.x,features_train),1)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()



        # img_pth = os.path.join(self.image_root, self.maskroot + self.data.image_id[idx])
        # #print(img_pth)
        # img = Image.open(img_pth).convert("RGB")

        # if self.transform:
        #     img = self.transform(img)
        #  no transform for tensor!


       

        label = self.data.label[idx]
        label = self.label_dict[label]

        return self.x[idx], label

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