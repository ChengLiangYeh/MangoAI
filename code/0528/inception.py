import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from MangoDataset0528 import *
import matplotlib.pyplot as plt
import torchvision.models as models
import os
import sys
import random

crossloss = nn.CrossEntropyLoss()

def check_accuracy(model, device,loader):
    model.eval()
    train_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            img = data.to(device)
# label = label.to(device, dtype=torch.int64    
            label = label.to(device)
            outputs, aux_outputs= model(img)
            # print(pre_label)
            loss1 = crossloss(outputs, label)
            
            loss2 = crossloss(aux_outputs, label)
            loss = loss1 + 0.4*loss2
           
            train_loss += loss.item()

    train_loss = train_loss / len(loader.dataset)
    return train_loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        img = data.to(device)
        # label = label.to(device, dtype=torch.int64)
        label = label.to(device)
        optimizer.zero_grad()

        outputs, aux_outputs= model(img)
        # print(pre_label)
        loss1 = crossloss(outputs, label)
        
        loss2 = crossloss(aux_outputs, label)
        loss = loss1 + 0.4*loss2
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    print('epoch {}：  {:.8f}'.format(epoch, train_loss))
    return train_loss

def main(size, rate):
    torch.cuda.empty_cache()
    print('size: ', size)
    print('rate: ', rate)
    pth_name = '0528/test-inception' + str(size) + '_' + str(rate)
    batch_size = size # 512, 100-1000
    learning_rate = rate  * 1e-5 # 1e-3, 1e-5-1e-2

    transform = transforms.Compose([
        transforms.Resize([299, 299]),
        Rotate(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = Mango(data='train', transform=transform)
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [4760, 840])  #split train_data into train and val 
    print('train:', len(dataset_train), 'validation:', len(dataset_val))
    
    #print("dataset",dataset)
    #random.shuffle(dataset)

    #print("dataset",dataset)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=6)


    device = torch.device("cpu")
    print('device: ', device)

    model =  models.inception_v3(pretrained=True)
 

    model.AuxLogits.fc = nn.Linear(768, 3)
    model.fc = nn.Linear(2048, 3)

    # freezing
    ct = 0
    for child in model.children():
        ct += 1
        if ct <1:
            for param in child.parameters():
                param.requires_grad = False
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    print(model)

    plt.figure()
    train_losses = []
    val_losses =[]
    epochs = []
    count = 0
    best_loss = np.finfo('f').max
    for epoch in range(50):
        print("start train....")
        train_loss = train(model, device, train_loader, optimizer, epoch)

        val_loss = check_accuracy(model,device,val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch)
        if val_loss < best_loss:  #用validation loss 來看有沒有overfit，如果val loss變大，但是train loss還是變小，代表overfit了，因此就early stop
            best_loss = val_loss
            count = 0
        else:
            count += 1
            if count > 10:  #超過10次 val loss > best loss
                print("early stop\n\n")
                break
        print('      count = ', count)

    # plt.plot(epochs, train_losses, label='training loss')
    # plt.plot(epochs, val_losses, label='validation loss')
    # plt.legend(loc='upper right')
    torch.save(model.state_dict(), pth_name + '.pth')
    # pth_name = pth_name + '_' + str(int(10*best_loss))
    # plt.savefig('./result_pics/' + pth_name + '.png')
    

if __name__ == '__main__':
    # size = np.random.randint(low=100, high=1000)
    # rate = np.random.randint(low=1, high=100)
    # sizes = [152, 161, 236, 252, 278, 374, 494, 575, 588, 642]
    # rates = [1, 21, 41, 61, 81, 101]
    # for i in range(6):
    size = 64
    rate = 70
    main(size, rate)