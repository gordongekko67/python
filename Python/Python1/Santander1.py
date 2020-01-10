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

# lettura dei due files
df = pd.read_csv("/Users/Enrico/Downloads/santander-customer-transaction-prediction/train.csv")
dl = pd.read_csv("/Users/Enrico/Downloads/santander-customer-transaction-prediction/test.csv")

tes1 = dl['ID_code']

# tolgo le colonne a x_train
x_train = df.drop('ID_code', 1)
x_train = x_train.drop('target',1 )

# tolgo le colonne a x_test
x_test = dl.drop('ID_code', 1)

print (x_train.head)
# ricavo Y_train
y_train = df['target']
print (y_train.head)

#normalizzare x.train
df_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_train1 = df_scaler.fit_transform(x_train)
x_test1  = df_scaler.fit_transform(x_test)

#normalizzare y.train
y_train1 = np_utils.to_categorical(y_train, 2)

# modello neurale
model = Sequential()
model.add(layers.Dense(200, input_dim=x_train1.shape[1], activation='relu'))
model.add(layers.Dense(150, activation = 'relu'))
model.add(layers.Dense(130, activation = 'relu'))
model.add(layers.Dense(100, activation = 'relu'))
model.add(layers.Dense(80, activation = 'relu'))
model.add(layers.Dense(60, activation = 'relu'))
model.add(layers.Dense(40, activation = 'relu'))
model.add(layers.Dense(30, activation = 'relu'))
model.add(layers.Dense(15, activation = 'relu'))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(2, activation = 'softmax'))

print (model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])

history = model.fit(x_train1,y_train1, batch_size=128, epochs=10, verbose = 1, validation_split=0.0
                    )
pred = model.predict(x_test1)
print (len(pred))
print (pred.shape)

ynew = model.predict_classes(x_test1)
# show the inputs and predicted outputs
for i in range(len(x_test1)):
	print("Predicted=%s" % ( ynew[i]),i  )

# caloolo lo score
score = model.evaluate(x_test1, pred, verbose = 0)
print (" score 2 " +  score)
# creo un nuovo data frame e vado su csv
pred_dataframe = pd.DataFrame(ynew)
print(pred_dataframe)
result = pd.concat([tes1, pred_dataframe], axis=1)
result.columns = ['ID_CODE', 'target']
result.to_csv("santander.csv",index=False)

