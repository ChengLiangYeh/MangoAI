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

if __name__ == '__main__':
    features_train = torch.load('ptfile/vgg16.pt')
    data_train = pd.read_csv('C1-P1_Train Dev_fixed/' + 'train.csv').iloc[:, 1].values
    features_dev = torch.load('ptfile/vgg16dev_.pt')
    data_dev = pd.read_csv('C1-P1_Train Dev_fixed/' + 'dev.csv').iloc[:, 1].values
    label_train = []
    label_dev = []
    for label in data_train:
        if label == 'A':
            label_train.append(1)
        elif label == 'B':
            label_train.append(2)
        else:
            label_train.append(3)
    for label in data_dev:
        if label == 'A':
            label_dev.append(1)
        elif label == 'B':
            label_dev.append(2)
        else:
            label_dev.append(3)

    features_train = features_train.numpy()
    features_dev = features_dev.numpy()

    X = features_train
    Y = label_train
    print(X.shape)
    logreg = LogisticRegression(C=4)
    logreg.fit(X, Y)


    score = logreg.score(X, Y)
    print("train_score",score)
    score = logreg.score(features_dev,label_dev )
    print("test_score",score)

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