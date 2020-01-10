import numpy as np
from keras import models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import matplotlib.pyplot  as plt

#imatplotlib inline
np.set_printoptions(linewidth=1300)

from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers  import Dense, activations
from keras import layers

from keras.optimizers  import SGD
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#lettura file da directory o da internet
df = pd.read_csv("/Users/Enrico/pid.csv")
#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header = 0)

# verifica se esistono dati nulli
print (df.isnull().sum())

#elimino una colonna che non mi serve
df = df.drop(['Unnamed: 0'], axis = 1)

print (df.head)
# separo la label in x_train e y_train
df, cl = (df.iloc[:, 0:8], df['diabetes'])

# pongo tutto nella stessa scala  x_train e y_train
df_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df1 = df_scaler.fit_transform(df)
cl1 = LabelEncoder().fit_transform(cl)

# spezzo il database in train e test
x_train, x_test, y_train, y_test = train_test_split(df1, cl1, test_size=0.3)

# Trasformo y_train , e y_test in variabile dummy
y_train = np_utils.to_categorical(y_train,2 )
y_test  = np_utils.to_categorical(y_test ,2 )
#
# modello neurale
model1 = Sequential()
model1.add(layers.Dense(50,input_dim=x_train.shape[1] , activation='relu'))
model1.add(layers.Dense(40, activation = 'relu'))
model1.add(layers.Dense(30, activation = 'relu'))
model1.add(layers.Dense(20, activation = 'relu'))
model1.add(layers.Dense(10, activation = 'relu'))
model1.add(layers.Dense(2, activation = 'sigmoid'))

print (model1.summary())

model1.compile(loss = 'binary_crossentropy', optimizer='adam', metrics= ['accuracy'])
history = model1.fit(x_train, y_train, batch_size=10, epochs=50, verbose = 1 , validation_split=0.3)
score = model1.evaluate(x_test,y_test, verbose =1 )
print (score)

