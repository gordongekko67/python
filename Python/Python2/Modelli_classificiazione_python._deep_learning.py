import pandas as pd
import numpy as np
from keras.utils  import np_utils

import keras
from keras.models import Sequential
from keras import layers
from keras.optimizers  import SGD

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = 0)
df = df.drop('ID_code', 1)
df = df.drop('target')
print (df.head())
df.columns = ['sep_len  ', 'sep_wid  ', 'pet_len  '  , ' pet_wid  ', 'Class']
print (df.head())

df, cl = (df.iloc[:, 0:4], df['Class'])
print (df.head(5))
print (cl.head(5))

print (cl.value_counts())

#normalizzare df.train
df_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df1 = df_scaler.fit_transform(df)

print (df1)
print (df1.shape)


#normalizzare cl.train
cl1 = LabelEncoder().fit_transform(cl)
print (cl1)

x_train, x_test, y_train, y_test = train_test_split(df1, cl1, test_size=0.3)
print (x_train.shape, x_test.shape)

y_train = np_utils.to_categorical(y_train, 3)
y_test  = np_utils.to_categorical(y_test,3 )

print (y_train, y_test)

# modello neurale
model = Sequential()

model.add(layers.Dense(50, input_dim=x_train.shape[1], activation='relu'))
model.add(layers.Dense(40, activation = 'relu'))
model.add(layers.Dense(30, activation = 'relu'))
model.add(layers.Dense(25, activation = 'relu'))
model.add(layers.Dense(3, activation = 'softmax'))

print (model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

history = model.fit(x_train, y_train, batch_size=10, epochs=50, verbose = 1 , validation_split=0.3)

score = model.evaluate(x_test,y_test)
print (score)

# secondo dataset
df = pd.read_csv('/Users/enrico/pid.csv', header = 0)
df = df.drop(['Unnamed: 0'], axis = 1)
print (df.head(5))

df, cl = (df.iloc[:, 0:8], df['diabetes'])
print (df.shape)
#normalizzare dataset
df_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df1 = df_scaler.fit_transform(df)

#normalizzare cl.train
cl1 = LabelEncoder().fit_transform(cl)

x_train, x_test, y_train, y_test = train_test_split(df1, cl1, test_size=0.3)
print (x_train.shape, x_test.shape, y_train.shape, y_test.shape)

y_train = np_utils.to_categorical(y_train, 2)
y_test  = np_utils.to_categorical(y_test,  2)

# modello neurale
model1 = Sequential()

model1.add(layers.Dense(50, input_dim=x_train.shape[1], activation='relu'))
model1.add(layers.Dense(40, activation = 'relu'))
model1.add(layers.Dense(30, activation = 'relu'))
model1.add(layers.Dense(25, activation = 'relu'))
model1.add(layers.Dense(2, activation = 'sigmoid'))

print (model1.summary())

model1.compile(loss = 'binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

history = model1.fit(x_train, y_train, batch_size=10, epochs=50, verbose = 1 , validation_split=0.2)

score1 = model1.evaluate(x_test,y_test, verbose = 1)
print (score1)


pred = model1.predict(x_test)
print(pred)
