import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
from MangoDataset import *

if __name__ == '__main__':
    model = models.alexnet(pretrained=True)
    print(model)
    model.fc = nn.Sequential(nn.Linear(1000, 100),
                            nn.ReLU(),
                            nn.Linear(100, 10),
                            nn.ReLU(),
                            nn.Linear(10, 3),
                            nn.Softmax(dim=3))
    # model.fc = nn.Linear(100, 10)
    # model.fc = nn.Linear(10, 3)
    # vgg16 = models.vgg16(pretrained=True)
    # densenet = models.densenet161(pretrained=True)
    # shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
    # resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
    print(model)