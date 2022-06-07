import pandas as pd
import numpy as np


if __name__ == '__main__':
	csv_list = ['densenet161', 'resnext', 'squeezenet', 'vgg16']
	label2num = {'A':0, 'B':1, 'C':2}
	num2label = {0:'A', 1:'B', 2:'C'}
	datas = []
	for csv in csv_list:
		data = pd.read_csv('model output csv/' + csv + '-bagging-result.csv', sep=',',header=None)
		datas.append(data)
	# print(label_dict[datas[0].iloc[0].values[0]])
	output = []
	for i in range(1600):
		vote = np.array([0, 0, 0])
		for model in datas:
			decision = label2num[model.iloc[i].values[0]]
			vote[decision] += 1

		output.append(num2label[np.random.choice(np.flatnonzero(vote == vote.max()))])

	print(output)
	import csv
	with open('bagging-result.csv', 'a', newline='') as csvfile:
		writer = csv.writer(csvfile)
		for i in output:
			writer.writerow(i)
