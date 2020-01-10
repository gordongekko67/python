import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot  as plt

#imatplotlib inline

from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.layers  import Dense, activations
from keras import layers

from keras.optimizers  import SGD
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
eti = set (y_train)
print (eti)

plt.figure(figsize = ( 16, 16))

i = 1
for  e in eti :
      image = x_train[y_train.tolist().index(e)]
      plt.subplot(10, 10, i)
      plt.axis('off')
      #plt.title()
      i +=1
      _=plt.imshow(image, cmap = 'gray')

plt.show()

print (x_train.shape)

x_train = np.reshape(x_train, (60000, 784))
x_test  = np.reshape(x_test, (10000, 784))

print (x_train.shape)
print (x_test.shape)

print('valori x_train')
print (x_train)
x_train = x_train.astype('float32')/255
print (x_train)
x_test = x_test.astype('float32')/255

y_train= np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


#model1

model1 = Sequential()
model1.add(Dense(10,  input_shape = (784,) ))
#model1.add(Activation('softmax'))

model1.summary()

model1.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
history = model1.fit(x_train, y_train, batch_size=128, epochs=10, verbose = 1 , validation_split=0.2)

score = model1.evaluate(x_test, y_test, verbose= 1)
print(score)


#model2

model2 = Sequential()
model2.add(Dense(700,  input_dim = 784,  activation='relu' ))
model2.add(Dense(700,   activation='relu' ))
model2.add(Dense(350,   activation='relu' ))
model2.add(Dense(100,   activation='relu' ))
model2.add(Dense(10,   activation='softmax' ))
model2.summary()

model2.compile(loss = 'categorical_crossentropy', optimizer='SGD', metrics= ['accuracy'])
history = model2.fit(x_train, y_train, batch_size=128, epochs=10, verbose = 1 , validation_split=0.2)

score2 = model2.evaluate(x_test, y_test, verbose= 1)
print(score2)
#-------------------------------------------------------------------------
# ridimensionamento
image_height, image_widht = 28, 28
x_train = x_train.reshape(x_train.shape[0], image_height, image_widht, 1 )
x_test  = x_test.reshape(x_test.shape[0], image_height, image_widht, 1 )
input_shape = (image_height, image_widht, 1)
print (x_train.shape)

#model3
print ('----------------------------model3---------------------------------------------')
model3 = Sequential()
model3.add(layers.Conv2D(64,  kernel_size=(3,3),  activation='relu', input_shape=input_shape ))
model3.add(layers.MaxPooling2D(pool_size=(2,2)))
model3.add(layers.Conv2D(128, kernel_size=(3,3),  activation='relu', padding ='same' ))
model3.add(layers.MaxPooling2D(pool_size=(2,2)))
model3.add(layers.Dropout(0.5))
model3.add(layers.Flatten())
model3.add(Dense(128,   activation='relu' ))
model3.add(layers.Dropout(0.5))
model3.add(Dense(10,   activation='softmax' ))

model3.summary()

model3.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
history = model3.fit(x_train, y_train, batch_size=128, epochs=10, verbose = 1 , validation_split=0.2)

#Rete convonuzionale con Python MNIST.py


pred = model3.predict(x_test)
