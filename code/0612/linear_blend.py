from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from blend_loader import *
from model import *
from plot_utils import *
from torchsummary import summary
#import geffnet
from engine import *

from torch.utils.data.sampler import  RandomSampler #0611

class TwoLayerNet(torch.nn.Module):
    def __init__(self,d):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(d, 1)
        self.linear2 = torch.nn.Linear(d, 1)
        self.linear3 = torch.nn.Linear(d, 1)
        self.d = d

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
       
        a = x.view(len(x),self.d,3)
        a_t = torch.transpose(a,1,2)
        pred_1 =self.linear1(a_t[:,0])
        pred_2 =self.linear2(a_t[:,1])
        pred_3 =self.linear3(a_t[:,2])
        blend = torch.cat((pred_1,pred_2,pred_3),1)
        return blend


# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.

# Create random Tensors to hold inputs and outputs
x = torch.randn(10, 12)
print(x)
# Construct our model by instantiating the class defined above
model = TwoLayerNet(9)
model = model.to("cpu")
device=torch.device('cpu')
# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.



batch_size = 40
lr_rate = 0.0001
num_epochs = 5000
momentum = 0.9
name_list =["all .pt/alexnetdev_0611.pt","all .pt//densenet121dev_0611.pt","all .pt//densenet161dev_0611.pt","all .pt/resnetdev_0611.pt"
,"all .pt/resnextdev_0611.pt","all .pt/vgg11dev_0611.pt","all .pt/vgg16dev_0611.pt"]
dataset = Mango(name_list)
indices = torch.randperm(len(dataset)).tolist()
dataset_train = torch.utils.data.Subset(dataset, indices[:-200])
dataset_test = torch.utils.data.Subset(dataset, indices[-200:])

dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
dataloader_test = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)
# iterator = cycle(dataloader)

dataloaders = {'train':dataloader_train, #0611
                    'val':dataloader_test}

optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9,weight_decay=0.5)
criterion = nn.CrossEntropyLoss()
model_name = "linear_blend"

model_best, hist, best_acc , epoch_train_loss_history, epoch_val_loss_history = train_model(model, dataloaders, criterion, optimizer, device, lr_rate ,num_epochs=num_epochs, is_inception=False) 
    #torch.save(model_best.state_dict(), 'model_dict/' + model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '.pth')
torch.save(model_best.state_dict(), 'model_dict/' + model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging' + '.pth')
    #plot_models_loss(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size), hist, model_name)
plot_models_loss(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging', hist, model_name)
plot_loss_train_and_val(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging',epoch_train_loss_history,epoch_val_loss_history)


