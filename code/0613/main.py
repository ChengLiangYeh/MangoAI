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
#import geffnet
from engine import *
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.service.ax_client import AxClient
from adabound import AdaBound
from torch.utils.data.sampler import  RandomSampler #0611

def main(parameters):
    feature_extract = parameters.get('feature_extract', True)
    model_name = parameters.get('model_name', 'resnext')
    batch_size = parameters.get('batch_size', 32)
    #lr_rate = parameters.get('lr', 0.001)
    #lr_rate = parameters.get('lr', 0.009504895365762002)
    lr_rate = parameters.get('lr', 0.005)
    #momentum = parameters.get('momentum', 0.31507929041981697)
    momentum = parameters.get('momentum', 0.9)
    num_epochs = parameters.get('num_epochs', 100)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)
    dataset_train = get_mango_dataset(data='train', mask=True, transform='default')
    dataset_dev = get_mango_dataset(data='dev', mask=True, transform='default')
    sampler = RandomSampler(dataset_train,replacement=True)#0611

    #dataloaders = {'train':DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
    #                'val':DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=6)}
    dataloaders = {'train':DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=6), #0611
                    'val':DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=6)}

    #model = initialize_model(model_name, 3, feature_extract, use_pretrained=True)
    model = initialize_model(model_name, 3, feature_extract, use_pretrained=True)
    model = model.to(device)
    #解放層
    #unfreeze_layers(model, 4)
    unfreeze_total_layers(model)

    # # for efficientnet, under construction
    # model = geffnet.create_model('tf_efficientnet_b7_ns', pretrained=True)
    #model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)######
    #model = model.to(device)######
    # set_parameter_requires_grad(model, feature_extract)
    # unfreeze_layers(model, 2)

    #summary(model, (3, 224, 224)) #inception = 299
    #summary(model, input_size=(3, 224, 224))
    print('\nhyper-parameters:\n', parameters)

    params_to_update = get_param_to_update(model, feature_extract)  
    optimizer = optim.SGD(params_to_update, lr=lr_rate, momentum=momentum)
    #optimizer = AdaBound(params_to_update, lr_rate, betas=(0.9, 0.999), final_lr=0.1, gamma=0.001, weight_decay=0.0005)
    #optimizer = optim.Adam(params_to_update, lr=lr_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    #model_best, hist, best_acc , epoch_train_loss_history, epoch_val_loss_history = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=num_epochs,is_inception=True) #inception!!!
    ###model_best, hist, best_acc , epoch_train_loss_history, epoch_val_loss_history = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=num_epochs)
    model_best, hist, best_acc , epoch_train_loss_history, epoch_val_loss_history = train_model(model, dataloaders, criterion, optimizer, scheduler, device, lr_rate ,num_epochs=num_epochs, is_inception=False) #加了調整lr
    #torch.save(model_best.state_dict(), 'model_dict/' + model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '.pth')
    torch.save(model_best.state_dict(), 'model_dict/' + model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging' + '.pth')
    #plot_models_loss(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size), hist, model_name)
    plot_models_loss(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging', hist, model_name)
    plot_loss_train_and_val(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '_unfreeze-total' + '_momentum' + str(momentum) + '_sharp + blur data + input_size=224_testbagging',epoch_train_loss_history,epoch_val_loss_history)

    return best_acc.item()


if __name__ == '__main__':

    
    # check model.py for available models
    parameter = {
        'model_name':'mobilenet',
        #'lr':0.009504895365762002,
        'lr':0.00001,
        #'momentum':0.31507929041981697,
        'momentum':0.9,
        'num_epochs':100,
    }
    best_acc = main(parameter)

    ''' # for find hyperparameter
    ax_client = AxClient()
    ax_client.create_experiment(
        name="Mango_experiment",
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-5, 0.04], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
            ],
    )
    parameters, trial_index = ax_client.attach_trial(parameters={"lr": 0.005, "momentum": 0.9})
    ax_client.complete_trial(trial_index=trial_index, raw_data=main(parameters))
    for i in range(20):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(trial_index=trial_index, raw_data=main(parameters))
    best_parameters, values = ax_client.get_best_parameters()
    print('best_parameters=',best_parameters, 'value=',values)

    f = open('trainlog', 'a')
    f.write('best_parameters=' + str(best_parameters) + '\n')
    f.write('value=' + str(values) + '\n')
    f.write('\n')
    '''