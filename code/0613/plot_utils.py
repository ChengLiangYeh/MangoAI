import torch
import matplotlib.pyplot as plt
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0.0001):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.adjust_lr_times = 0  ###counter = 9 => 調整lr變小 (調整3次)

    def __call__(self, val_loss, lr_rate): ###

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score > self.best_score - self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            ###counter = 9 => 調整lr變小 (調整3次) => 好像scheduler就有調了？！ => 不用加(0608已確認) => 加了效果也不會變好!
            ###if self.counter == 9 and self.adjust_lr_times < 3:
                ###print('lr_rate=',lr_rate)
                ###if  self.adjust_lr_times == 0:
                    ###lr_rate = lr_rate / 2
                ###elif self.adjust_lr_times == 1:
                    ###lr_rate = lr_rate / 4
                ###elif self.adjust_lr_times == 2:
                    ###lr_rate = lr_rate / 8
                ###elif self.adjust_lr_times == 3:
                    ###lr_rate = lr_rate / 120
                ###elif self.adjust_lr_times == 4:
                    ###lr_rate = lr_rate / 720

                ###print('lr_rate(after ajustment)=',lr_rate)
                ###self.counter = 0
                ###self.adjust_lr_times += 1
                ###print('adjust_lr_times(max is 3)=',self.adjust_lr_times)
                

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def plot_loss_train_and_val(title,epoch_train_loss_history,epoch_val_loss_history):
    plot2 = plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    y1 = epoch_train_loss_history
    y2 = epoch_val_loss_history
    x = range(len(y1))
    plt.plot(x,y1)
    plt.plot(x,y2)
    plot2.savefig('result_pics/' + title + 'loss(train+val)' + '.png')


def plot_models_loss(title, *args, xlabel='Training Epochs', ylabel='Validation Accuracy'):
    plot = plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    for i in range(0,len(args),2):
        hist = args[i]
        hist = [h.cpu().numpy() for h in hist]
        label = args[i+1]
        plt.plot(range(1,len(hist)+1), hist,label=label)

    plt.legend()
    plot.savefig('result_pics/' + title + '.png')

if __name__ == '__main__':
    device = torch.device("cuda:0")
    hist1 = torch.tensor([1, 2, 6, 7, 2, 3, 4, 5, 7, 4]).to(device)
    hist2 = torch.tensor([9, 7, 0, -1, 17, 22, 3, 3, 8, 7, 8, 4, 1, -5]).to(device)
    hist3 = torch.tensor([21, 15, 3, 10, 5]).to(device)
    title = 'test'
    plot_models_loss(title, hist1, 'one', hist2, 'two', hist3, 'three')
