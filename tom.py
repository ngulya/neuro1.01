import pandas as pd
import numpy as np
import os
from keras.models import model_from_json
from keras import backend as K

hist = pd.read_csv("./tomorrow.csv")
hist['Weekday'] =  pd.DataFrame({'Weekday':[]})
hist['Week'] =  pd.DataFrame({'Week':[]})
hist['Date'] = pd.to_datetime(hist['Date'])
hist['Week'] = hist['Date'].dt.week
hist['Weekday'] = hist['Date'].dt.weekday
price = []
predict = []
n = 0
n2 = 0
while n2 < 2:
	price.append([])
	n = 0
	while n < 30:
		price[n2].append(hist['High'][n + n2]/ 100000)
		price[n2].append(hist['Close'][n + n2]/ 100000)
		price[n2].append(hist['Open'][n + n2]/ 100000)
		price[n2].append(hist['Low'][n + n2]/ 100000)
		price[n2].append(hist['Week'][n + n2] / 53)
		price[n2].append(hist['Weekday'][n + n2] / 6)
		n = n + 1
	n2 = n2 + 1
price = np.asarray(price)
if os.path.exists("./model.json") and os.path.exists("./model.h5"):
	json_file = open('model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	model.load_weights("model.h5")
	model.compile(loss="mean_squared_error", metrics=["accuracy"], optimizer="Adam")
	resul = model.predict(price)
	print('\nHigh	',resul[0][0]*100000)
	print('Close	',resul[0][1]*100000)
	print('Open	',resul[0][2]*100000)
	print('Low	',resul[0][3]*100000)
K.clear_session()