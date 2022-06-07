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
import torchvision.models as models
import os
import sys

crossloss = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        img = data.to(device)
        # label = label.to(device, dtype=torch.int64)
        label = label.to(device)
        optimizer.zero_grad()

        pre_label = model(img)
        loss = crossloss(pre_label, label)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    print('epoch {}：  {:.8f}'.format(epoch, train_loss))
    return train_loss

def main(size, rate):
    print('size: ', size)
    print('rate: ', rate)
    pth_name = '0520/test' + str(size) + '_' + str(rate)
    batch_size = size # 512, 100-1000
    learning_rate = rate * 2 * 1e-5 # 1e-3, 1e-5-1e-2

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        Rotate(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    dataset = Mango(data='train', transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    model = models.alexnet(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(1000, 100),
                            nn.ReLU(),
                            nn.Linear(100, 10),
                            nn.ReLU(),
                            nn.Linear(10, 3),
                            nn.Softmax(dim=3))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)
    print(model)

    plt.figure()
    train_losses = []
    epochs = []
    count = 0
    best_loss = np.finfo('f').max
    for epoch in range(100):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        epochs.append(epoch)
        if train_loss < best_loss:
            best_loss = train_loss
            count = 0
        else:
            count += 1
            if count > 5:
                print("early stop\n\n")
                break
        print('      count = ', count)

    plt.plot(epochs, train_losses, label='total loss')
    plt.legend(loc='upper right')
    torch.save(model.state_dict(), './model_dict/' + pth_name + '.pth')
    pth_name = pth_name + '_' + str(int(10*best_loss))
    plt.savefig('./result_pics/' + pth_name + '.png')
    

if __name__ == '__main__':
    # size = np.random.randint(low=100, high=1000)
    # rate = np.random.randint(low=1, high=100)
    # sizes = [152, 161, 236, 252, 278, 374, 494, 575, 588, 642]
    # rates = [1, 21, 41, 61, 81, 101]
    # for i in range(6):
    size = 128
    rate = 27
    main(size, rate)