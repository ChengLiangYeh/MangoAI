from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import copy
from MangoDataset import *
from model import *
from plot_utils import *
# from efficientnet_pytorch import EfficientNet
from torchsummary import summary
# import geffnet

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, best_acc

if __name__ == '__main__':
    feature_extract = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    batch_size = 32
    dataset_train = get_mango_dataset(data='train', mask=True, transform='default')
    dataset_dev = get_mango_dataset(data='dev', mask=True, transform='default')
    dataloaders = {'train':DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
                    'val':DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=6)}
    
    
    model = initialize_model('squeezenet', 3, feature_extract, use_pretrained=True)
    # model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)
    # model = geffnet.create_model('tf_efficientnet_b7_ns', pretrained=True)
    model = model.to(device)

    set_parameter_requires_grad(model, feature_extract)
    unfreeze_layers(model, 2)
    summary(model, input_size=(3, 224, 224))
    params_to_update = get_param_to_update(model, feature_extract)
    

    # Observe that all parameters are being optimized
    optimizer = optim.RMSprop(params_to_update, lr=0.002, momentum=0.9, weight_decay=0.9, eps=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_best, hist = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=15)
    torch.save(model_best.state_dict(), 'efficientnet-b7_10_0530.pth')
    

    # Initialize the non-pretrained version of the model used for this run
    # fine_model = initialize_model('squeezenet', 3, feature_extract=False, use_pretrained=True)
    # fine_model = fine_model.to(device)
    # fine_params_to_update = get_param_to_update(fine_model, False)
    # fine_optimizer = optim.SGD(fine_params_to_update, lr=0.001, momentum=0.9)
    # fine_criterion = nn.CrossEntropyLoss()
    # fine_model_best, fine_hist = train_model(fine_model, dataloaders, fine_criterion, fine_optimizer, num_epochs=2)
    # torch.save(fine_model_best.state_dict(), 'squeezenet_fine_tuning_2.pth')

    