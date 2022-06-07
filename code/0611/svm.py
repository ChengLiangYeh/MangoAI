import torch
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix

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
if __name__ == '__main__':
    features_train = torch.load('feature/vgg16train_.pt')
    data_train = pd.read_csv('C1-P1_Train Dev_fixed/' + 'train.csv').iloc[:, 1].values
    features_dev = torch.load('feature/vgg16dev_.pt')
    data_dev = pd.read_csv('C1-P1_Train Dev_fixed/' + 'dev.csv').iloc[:, 1].values
    label_train = []
    label_dev = []
    for label in data_train:
        if label == 'A':
            label_train.append(0)
        elif label == 'B':
            label_train.append(1)
        else:
            label_train.append(2)
    for label in data_dev:
        if label == 'A':
            label_dev.append(0)
        elif label == 'B':
            label_dev.append(1)
        else:
            label_dev.append(2)

    features_train = features_train.numpy()
    features_dev = features_dev.numpy()

    X = features_train
    Y = label_train
    print(X.shape)
    # SVM = S(C=4,multi_class='multinomial')
    Clist = [0.0001,0.001,0.1,1,10,100]
    kernelname = ['linear','poly','rbf','sigmoid']
    degree_list = [1,2,3,4,5,6]
    gamma_list = ['scale',0,1,2]
    for C in Clist:
        for kernel in kernelname:
            for degree in degree_list:
                for gam in gamma_list:
                    print("model: ",kernel)
                    print("C,degree,gamma",C,degree,gam)
                    print(".........")
                    SVM =SVC(C=C, kernel=kernel, degree=degree, gamma=gam, coef0=0.0,class_weight="balanced",probability=True)
                    SVM.fit(X,Y)

                    score = SVM.score(X, Y)
                    print("train_score",score)
                    score = SVM.score(features_dev,label_dev )
                    print("test_score",score)

    # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(SVM, features_dev,label_dev,
                                     cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # h = .02  # step size in the mesh
    # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    # plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # # Plot also the training points
    # plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    # plt.xlabel('Sepal length')
    # plt.ylabel('Sepal width')

    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.xticks(())
    # plt.yticks(())

    # plt.show()