import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
import datetime
import copy
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
import scipy.sparse.csr as csr

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


hist = pd.read_csv("./history.csv")
hist['Weekday'] =  pd.DataFrame({'Weekday':[]})
hist['Week'] =  pd.DataFrame({'Week':[]})
hist['Day'] = pd.DataFrame({'Day':[]})

hist['Date'] = pd.to_datetime(hist['Date'])

hist['Week'] = hist['Date'].dt.week
hist['Weekday'] = hist['Date'].dt.weekday
hist['Day'] = hist['Date'].dt.dayofyear
# print(hist[0:15])

mas = []
n = 0
n2 = 0
# print(hist['High'][0])
while n2 < 2:
	mas.append([])
	n = 0
	while n < 30:
		mas[n2].append([])
		mas[n2][n].append(hist['High'][n + n2])
		mas[n2][n].append(hist['Close'][n + n2])
		mas[n2][n].append(hist['Open'][n + n2])
		mas[n2][n].append(hist['Low'][n + n2])
		mas[n2][n].append(hist['Week'])
		mas[n2][n].append(hist['Weekday'])
		mas[n2][n].append(hist['Day'])
		n = n + 1
	n2 = n2 + 1
print(mas[0][1][0:5])
