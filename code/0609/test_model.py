import torch
import pandas as pd
from torch import nn
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
from model import *
import csv
from scipy.special import softmax

if __name__ == '__main__':
    device = torch.device('cuda')

    model = initialize_model('densenet161', 3, True, use_pretrained=False)

    model.load_state_dict(torch.load('densenet161_lr0.009504895365762002_bsize32_epoch100_unfreeze-total_momentum0.31507929041981697_findhyperparameter.pth'))



    model.to(device)
    model.eval()
    data_dev = pd.read_csv('../MangoData/' + 'dev.csv').iloc[:, 1].values
    label_dev = []
    for label in data_dev:
        if label == 'A':
            label_dev.append(0)
        elif label == 'B':
            label_dev.append(1)
        else:
            label_dev.append(2)

    # features_train = features_train.numpy()
    # features_dev = features_dev.numpy()


    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.Grayscale(num_output_channels=3),
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
            images, labels, img_name = data
            images = images.to(device)
            

            outputs = model(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            # print("pre",predicted)
            # print("labels",labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            sort, idx = torch.sort(outputs.cpu(), descending=True)
            if((predicted == labels).item()!=1):
                a=predicted.numpy()
                b=labels.numpy()
                
                # print (type(a[0]))
                # print(type(img_name[0]))
                string = []
                string.append(img_name[0])
                string.append(str(predicted.numpy()[0]))
                string.append(str(labels.numpy()[0]))
                idx= idx.numpy()[0]
                sort = softmax(sort.numpy()[0])
                for i in range(3):
                    string.append(str(idx[i]))
                for i in range(3):
                    string.append(str(sort[i]))
                # string = img_name[0]+"pre "+str(a[0])+"label "+str(b[0])
                print(string)
                wrong_list.append(string)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    with open('densenet_output_clache.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(wrong_list)
    # print(wrong_list)
    print('Accuracy of the network on the 10000 test images: %.2f %%' % (
    100 * correct / total))
    print(confusion_matrix)

    # X = features_train
    # Y = label_train
    # print(X.shape)
    # # logreg = LogisticRegression(C=4,multi_class='multinomial')
    # logreg = LogisticRegression(C=5,multi_class='multinomial')
    # logreg.fit(X, Y)


    # score = logreg.score(X, Y)
    # print("train_score",score)
    # score = logreg.score(features_dev,label_dev )
    # print("test_score",score)

    # # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
    # titles_options = [("Confusion matrix, without normalization", None),
    #               ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(logreg, X,Y,
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp.ax_.set_title(title)

    #     print(title)
    #     print(disp.confusion_matrix)

    # plt.show()
    # # Plot the decision boundary. For that, we will assign a color to each
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