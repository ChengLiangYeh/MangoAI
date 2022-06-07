from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from MangoDataset import *
from model import *
from plot_utils import *
# from efficientnet_pytorch import EfficientNet
from torchsummary import summary
# import geffnet
from engine import *
# from ax.plot.trace import optimization_trace_single_method
# from ax.service.managed_loop import optimize
# from ax.utils.notebook.plotting import render

def main(parameters):
    feature_extract = parameters.get('feature_extract', True)
    model_name = parameters.get('model_name', 'squeezenet')
    batch_size = parameters.get('batch_size', 32)
    lr_rate = parameters.get('lr', 0.005)
    momentum = parameters.get('momentum', 0)
    num_epochs = parameters.get('num_epochs', 15)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset_train = get_mango_dataset(data='train', mask=True, transform='default')
    dataset_dev = get_mango_dataset(data='dev', mask=True, transform='default')
    dataloaders = {'train':DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=6),
                    'val':DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=6)}
    
    
    model = initialize_model(model_name, 3, feature_extract, use_pretrained=True)
    model = model.to(device)

    # # for efficientnet, under construction
    # model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=3)
    # model = geffnet.create_model('tf_efficientnet_b7_ns', pretrained=True)
    # model = model.to(device)
    # set_parameter_requires_grad(model, feature_extract)
    # unfreeze_layers(model, 2)

    summary(model, input_size=(3, 224, 224))
    print('\nhyper-parameters:\n', parameters)

    params_to_update = get_param_to_update(model, feature_extract)
    optimizer = optim.SGD(params_to_update, lr=lr_rate, momentum=momentum)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_best, hist, best_acc = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=num_epochs)
    torch.save(model_best.state_dict(), 'model_dict/' + model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size) + '_epoch' + str(num_epochs) + '.pth')
    plot_models_loss(model_name + '_lr' + str(lr_rate) + '_bsize' + str(batch_size), hist, model_name)

    return best_acc.item()


if __name__ == '__main__':
    # check model.py for available models
    parameter = {
        'model_name':'squeezenet',
        'lr':0.00579,
        'momentum':0.22426,
        'num_epochs':15,
    }
    best_acc = main(parameter)

    # best_parameters, values, experiment, model = optimize(
    #     parameters=[
    #         {"name": "lr", "type": "range", "bounds": [1e-4, 0.4], "log_scale": True},
    #         {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
    #         ],
    #     evaluation_function=main,
    #     total_trials=15,
    # )
    # print(best_parameters)
    # means, covariances = values
    # print(means)
    # best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
    # best_objective_plot = optimization_trace_single_method(
    #     y=np.maximum.accumulate(best_objectives, axis=1),
    #     title="Model performance vs. # of iterations",
    #     ylabel="Classification Accuracy, %",
    # )
    # render(best_objective_plot)

    