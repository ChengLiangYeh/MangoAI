from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from MangoDataset import *
from model import *
from plot_utils import *
from efficientnet_pytorch import EfficientNet
from torchsummary import summary
#from testaccuracyengine import *
from testengine_0612 import *
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.service.ax_client import AxClient
from adabound import AdaBound

def main(parameters):
    feature_extract = parameters.get('feature_extract', True)
    model_name = parameters.get('model_name', 'vgg16') #############################################################
    batch_size = parameters.get('batch_size', 32)
    lr_rate = parameters.get('lr', 0.005)
    momentum = parameters.get('momentum', 0.9)
    num_epochs = parameters.get('num_epochs', 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_train = get_mango_dataset(data='train', mask=True, transform='default')
    dataset_dev = get_mango_dataset(data='dev', mask=True, transform='default')
    dataset_test = get_mango_dataset(data='test', mask=True, transform='default')
    dataloaders = {'train':DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
                    'val':DataLoader(dataset_dev, batch_size=batch_size, shuffle=False, num_workers=6),
                    'test':DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=6)}
    
    model = initialize_model(model_name, 3, feature_extract, use_pretrained=True)
    ##########################################################
    model.load_state_dict(torch.load('./model_dict/before0611/vgg16_lr0.005_bsize32_epoch100_unfreeze-total_momentum0.9_sharp + blur data + input_size=224.pth'))
    ##########################################################
    model = model.to(device)


    params_to_update = get_param_to_update(model, feature_extract)  
    optimizer = optim.SGD(params_to_update, lr=lr_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    model_best, hist, best_acc , epoch_train_loss_history, epoch_val_loss_history = train_model(model, dataloaders, criterion, optimizer, scheduler, device, lr_rate ,num_epochs=num_epochs, is_inception=False)
    return best_acc.item()


if __name__ == '__main__':

    
    # check model.py for available models
    parameter = {
        'model_name':'vgg16',  #############################################################################
        #'lr':0.009504895365762002,
        'lr':0.005,
        #'momentum':0.31507929041981697,
        'momentum':0.9,
        'num_epochs':1,
    }
    best_acc = main(parameter)