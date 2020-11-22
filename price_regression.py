#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 11:19:58 2020

@author: specialist
"""

# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import train_test_split
import tensorflow
tensorflow.random.set_seed(100)
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Read the data
df = pd.read_csv('/Users/specialist/Documents/data/AB_US_2020.csv')

print('Available Features- ')
print(df.columns)
print('Missing Values- ')
print(df.isnull().sum())
print('Shape of Dataset- ', df.shape)

# Subset variables for ml
df1 = df.drop(['id', 'name', 'host_name', 'neighbourhood_group', 'latitude', 'longitude'], axis = 1)

# Convert categories
le = LabelEncoder()

for col in ['neighbourhood', 'room_type', 'city']:
    
    df1[col] = le.fit_transform(df1[col])

# Fill missing reviews with current date
df1['last_review'] = pd.to_datetime(df1['last_review'])
df1['last_review'] = df1['last_review'].fillna(pd.to_datetime('today'))

# Convert dates to unix continuous
df1['last_review'] = df1['last_review'].apply(lambda x: time.mktime(x.timetuple()))

# Fill missing
df1['reviews_per_month'] = df1['reviews_per_month'].fillna(0)

# Split data for modeling
y = df1['price']
x = df1.drop('price', axis = 1)

xtrain, xval, ytrain, yval = train_test_split(x, y, random_state = 100, test_size = .2)

# Build NN- https://towardsdatascience.com/regression-based-neural-networks-with-tensorflow-v2-0-predicting-average-daily-rates-e20fffa7ac9a
nn = Sequential()
nn.add(Dense(10, input_dim = 10, kernel_initializer = 'normal', activation = 'relu'))
nn.add(Dense(2700, activation = 'relu'))
nn.add(Dense(1, activation = 'linear'))
nn.summary()
nn.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse','mae'])
history = nn.fit(xtrain, np.array(ytrain), epochs = 30, batch_size = 10000, verbose = 1, validation_split = 0.2)
preds = nn.predict(xval)

# Visualize training over time
print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print('MAE for initial model is ', metrics.mean_absolute_error(yval, preds))

# Random Forest
rf = RandomForestRegressor(random_state = 100)
rf.fit(xtrain, ytrain)
rfpreds = rf.predict(xval)
print('MAE for RF model is ', metrics.mean_absolute_error(yval, rfpreds))
