import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from MangoDataset0530 import *
import torchvision.models as models
from torchsummary import summary
import numpy as np
import torch
import torchvision
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import torchvision.models as models
import os
import sys
import random
from model import*


crossloss = nn.CrossEntropyLoss()

def check_accuracy(model, device,loader):
    model.eval()
    train_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            img = data.to(device)
# label = label.to(device, dtype=torch.int64    
            label = label.to(device)
            pre_label = model(img)
            loss = crossloss(pre_label, label)
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

        pre_label = model(img)
        # print(pre_label)
        loss = crossloss(pre_label, label)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(train_loader.dataset)
    print('epoch {}：  {:.8f}'.format(epoch, train_loss))
    return train_loss

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes=10):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Linear(512+512, nb_classes)
        
    def forward(self, x):
        x1 = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        x1 = x1.view(x1.size(0), -1)
        x2 = self.modelB(x)
        x2 = x2.view(x2.size(0), -1)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)
        return x
def main(size, rate):
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    # map_location='cpu'
    # model = models.squeezenet1_0(pretrained=False)
    # num_classes =3
    # # set_parameter_requires_grad(model_ft, feature_extract)
    # model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
    # model.num_classes = num_classes
    # # model.load_state_dict('squeezenet_fine_tuning_10.pth')
    # model.load_state_dict(torch.load('squeezenet_fine_tuning_10.pth', map_location=map_location))
    # model.classifier[1] = nn.Identity()



    # print(model)
    # modules=list(model.features.children())[:-1]
    # resnet152=nn.Sequential(*modules)
    # for p in resnet152.parameters():
    #     p.requires_grad = False
    # model =resnet152
    # summary(model, input_size=(3, 224, 224))
    # print(model)



    modelA = initialize_model("squeezenet",3,True,use_pretrained = False)
    modelB = initialize_model("densenet161",3,True,use_pretrained = False)

    modelA.classifier[1] = nn.Identity()
    modelB.classifier = nn.Identity()

    # modelA.load_state_dict(torch.load("squeezenet_lr0.005_bsize128_epoch100unfreeze-total-momentum=0.9.pth"))
    # modelB.load_state_dict(torch.load("densenet161_lr0.005_bsize32_epoch100unfreeze-total-momentum=0.9.pth"))
    # Freeze these models


    # Create ensemble model
    model = MyEnsemble(modelA, modelB,3)
    model.load_state_dict(torch.load("fusion_train10_27.pth"))
    summary(model, input_size=(3, 224, 224))
    torch.cuda.empty_cache()
    print('size: ', size)
    print('rate: ', rate)
    pth_name = 'fusion_train' + str(size) + '_' + str(rate)
    batch_size = size # 512, 100-1000
    learning_rate = rate * 2 * 1e-5 # 1e-3, 1e-5-1e-2

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        Rotate(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    dataset_test = Mango(data='dev', transform=transform)
    testloader=DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=6)
    correct = 0
    total = 0
    wrong_list =[]


    nb_classes = 3
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data in testloader:
            images, labels,img_name = data

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # print("pre",predicted)
            # print("labels",labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if((predicted == labels).sum().item()!=1):
                a=predicted.numpy()
                b=labels.numpy()
                # print (type(a[0]))
                # print(type(img_name[0]))
                string = img_name[0]+"pre "+str(a[0])+"label "+str(b[0])
                print(string)
                wrong_list.append(string)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print(wrong_list)
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))
    print(confusion_matrix)

    # device = torch.device("cpu")
    # print('device: ', device)

    # model = models.vgg16(pretrained=True)
    # freezing
    # ct = 0
    # for child in model.children():
    #     ct += 1
    #     if ct <10:
    #         for param in child.parameters():
    #             param.requires_grad = False
    # model.classifier[6] = nn.Sequential(nn.Linear(4096, 3),
    #                         nn.Softmax(dim=1))
    
    # pth_name = pth_name + '_' + str(int(10*best_loss))
    # plt.savefig('./result_pics/' + pth_name + '.png')
    

if __name__ == '__main__':
    # size = np.random.randint(low=100, high=1000)
    # rate = np.random.randint(low=1, high=100)
    # sizes = [152, 161, 236, 252, 278, 374, 494, 575, 588, 642]
    # rates = [1, 21, 41, 61, 81, 101]
    # for i in range(6):
    size = 10
    rate = 27
    main(size, rate)
# Train your separate models
# ...



    





    # # testing
    # transform = transforms.Compose([
    #     transforms.Resize([224, 224]),
    #     Rotate(15),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])
    # dataset_test = Mango(data='dev', transform=transform)
    # testloader=DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=6)
    # correct = 0
    # total = 0
    # wrong_list =[]


    # nb_classes = 3
    # confusion_matrix = torch.zeros(nb_classes, nb_classes)
    # with torch.no_grad():
    #     for data in testloader:
    #         img_name,images, labels = data

    #         outputs = model(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         # print("pre",predicted)
    #         # print("labels",labels)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         if((predicted == labels).sum().item()!=1):
    #             a=predicted.numpy()
    #             b=labels.numpy()
    #             # print (type(a[0]))
    #             # print(type(img_name[0]))
    #             string = img_name[0]+"pre "+str(a[0])+"label "+str(b[0])
    #             print(string)
    #             wrong_list.append(string)
    #         for t, p in zip(labels.view(-1), predicted.view(-1)):
    #             confusion_matrix[t.long(), p.long()] += 1

    # print(wrong_list)
    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    # 100 * correct / total))
    # print(confusion_matrix)
