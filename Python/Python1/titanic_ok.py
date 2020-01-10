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

x_train = pd.read_csv("train.csv")
x_test  = pd.read_csv("test.csv")
############################################################################
####       estrarre una colonna
############################################################################
y_train = x_train.iloc[:,1].values
age = x_train.iloc[:,5].values
print (age)

############################################################################
# Trasformo sesso e embarked in numero
class_le= LabelEncoder()
x_train['Sex'] = class_le.fit_transform(x_train['Sex'].values)
x_test['Sex']  = class_le.fit_transform(x_test['Sex'].values)

emb_mapping = {'S':1, 'C': 2, 'Q':3 }
x_train['Embarked'] = x_train['Embarked'].map(emb_mapping)
x_test['Embarked'] = x_test['Embarked'].map(emb_mapping)
print (x_train)

# Trasformo y_train in variabile dummy
y_train1 = np_utils.to_categorical(y_train,2 )

# Normalizzazione  dei dati

# elimino una colonna
x_train = x_train.drop('Cabin', 1)
x_test = x_test.drop('Cabin', 1)
x_train =  x_train.drop('Ticket', 1)
x_test =  x_test.drop('Ticket', 1)
x_train =  x_train.drop('Name', 1)
x_test =  x_test.drop('Name', 1)

print('Errori')
print(x_train.isnull().sum())
from sklearn.preprocessing.imputation import Imputer

imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(x_train)
imputed_data = imr.transform(x_train.values)
print('trasf')
print (imputed_data[:200])

imr = Imputer(missing_values='NaN', strategy='mean', axis=1)
imr = imr.fit(x_test)
imputed_data2 = imr.transform(x_test.values)
print('trasf')
print (imputed_data2[:200])

std = StandardScaler()
x_train_std = std.fit_transform(imputed_data)
x_test_std = std.fit_transform(imputed_data2)

print (x_train_std)

print (imputed_data.shape)

#  rete neurale
# modello neurale
model1 = Sequential()

model1.add(layers.Dense(50, input_dim=9, activation='relu'))
model1.add(layers.Dense(40, activation = 'relu'))
model1.add(layers.Dense(30, activation = 'relu'))
model1.add(layers.Dense(25, activation = 'relu'))
model1.add(layers.Dense(2, activation = 'sigmoid'))

print (model1.summary())

model1.compile(loss = 'binary_crossentropy', optimizer='adam', metrics= ['accuracy'])

history = model1.fit(imputed_data, y_train1, batch_size=10, epochs=50, verbose = 1 , validation_split=0.3)

score1 = model1.evaluate(imputed_data,y_train1)

print (score1)


