import pandas as pd
import numpy as np
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.models import model_from_json
from keras.datasets import mnist
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

def to_array_data(size):

	price = []
	predict = []
	n = 0
	n2 = 0
	while n2 < size:
		price.append([])
		predict.append([])
		predict[n2].append(hist['High'][n2] / 100000)
		predict[n2].append(hist['Close'][n2] / 100000)
		predict[n2].append(hist['Open'][n2] / 100000)
		predict[n2].append(hist['Low'][n2] / 100000)
		n = 1
		while n < 31:
			price[n2].append(hist['High'][n + n2]/ 100000)
			price[n2].append(hist['Close'][n + n2]/ 100000)
			price[n2].append(hist['Open'][n + n2]/ 100000)
			price[n2].append(hist['Low'][n + n2]/ 100000)
			price[n2].append(hist['Week'][n + n2] / 53)
			price[n2].append(hist['Weekday'][n + n2] / 6)
			n = n + 1
		n2 = n2 + 1
	predict = np.asarray(predict)
	price = np.asarray(price)
	return price, predict


hist = pd.read_csv("./History.csv")
hist['Weekday'] =  pd.DataFrame({'Weekday':[]})
hist['Week'] =  pd.DataFrame({'Week':[]})
hist['Date'] = pd.to_datetime(hist['Date'])
hist['Week'] = hist['Date'].dt.week
hist['Weekday'] = hist['Date'].dt.weekday

price, predict = to_array_data(len(hist['Week']) - 30)

model = Sequential()
X_train, X_test, Y_train, Y_test = train_test_split(price, predict, test_size = 0.13)

if os.path.exists("./model.json") and os.path.exists("./model.h5"):
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")
else:
	model.add(Dense(180, input_dim = 180, activation="relu", kernel_initializer="normal"))
	model.add(Dense(90, activation="relu", kernel_initializer="normal"))
	model.add(Dense(45, activation="relu", kernel_initializer="normal"))
	model.add(Dense(4, activation="relu", kernel_initializer="normal"))

model.compile(loss="mean_squared_error", metrics=["accuracy"], optimizer="Adam")
print(model.summary())
y1 = input("Train model Y-Yes N-No: ")
if (y1 == 'y' or y1 == 'Y'):
	model.fit(X_train, Y_train, epochs = 2667, verbose = 1)
ans = model.predict(X_test)

print("{:.3f}".format(math.sqrt(mean_squared_error(Y_test, ans))))

price, predict = to_array_data(2)

size = 1
ans = model.predict(price)
ans = ans * 100000
predict = predict * 100000
print('	Must		Have')
while size >= 0:
	print('High	{0:.1f}		{1:.1f}'.format(predict[size][0], ans[size][0]))
	print('Close	{0:.1f}		{1:.1f}'.format(predict[size][1], ans[size][1]))
	print('Open	{0:.1f}		{1:.1f}'.format(predict[size][2], ans[size][2]))
	print('Low	{0:.1f}		{1:.1f}\n'.format(predict[size][3], ans[size][3]))
	size = size - 1

if(y1 == 'y' or y1 == 'Y'):
	yn = input("Save Neuro Y-Yes N-No: ")
	if (yn == 'y' or yn == 'Y'):
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("model.h5")
K.clear_session()
# 	Must		Have
# High	11501.4		11416.0
# Close	11359.4		11261.4
# Open	10903.4		10901.7
# Low	10639.8		10640.9

# High	11785.7		11684.8
# Close	11259.4		11372.7
# Open	11421.7		11527.4
# Low	11057.4		11083.7



# price = []
# predict = []
# n = 0
# n2 = 0
# while n2 < 2:
# 	price.append([])
# 	predict.append([])
# 	n = 0
# 	while n < 30:
# 		price[n2].append([])
# 		price[n2][n].append(hist['High'][n + n2])
# 		price[n2][n].append(hist['Close'][n + n2])
# 		price[n2][n].append(hist['Open'][n + n2])
# 		price[n2][n].append(hist['Low'][n + n2])
# 		price[n2][n].append(hist['Week'][n + n2])
# 		price[n2][n].append(hist['Weekday'][n + n2])
# 		n = n + 1
# 	predict[n2].append(hist['High'][n + n2])
# 	predict[n2].append(hist['Close'][n + n2])
# 	predict[n2].append(hist['Open'][n + n2])
# 	predict[n2].append(hist['Low'][n + n2])
# 	n2 = n2 + 1
# print(price[0][0:2])
# print(predict[0:2])

# model = Sequential()
# model.add(Conv2D(13, (6,6), input_shape = (30, 7, 1)))
# model.add(Activation('relu'))
# model.add(Conv2D(13, (2, 2)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dense(4))
# model.add(Activation('softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam')