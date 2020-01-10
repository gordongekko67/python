import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
#from scipy.misc import toimage

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

from keras.datasets  import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print (x_train.dtype)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print (x_train.dtype)
print (x_train)

x_train /= 255
x_test  /= 255


y_train= np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

for i in range (0,9):
    #image = x_train[y_train.tolist().index(i+1+330 )]
    plt.subplot(330 + 1 +i)
    #plt.imshow(toimage(x_train(i)))
    #_=plt.imshow(image, cmap = 'gray')

model = Sequential()
model.add(layers.Conv2D(32,  kernel_size=(3,3), padding = 'same', input_shape=(32,32,3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32,  kernel_size=(3,3),  padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),  padding = 'same'))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(64,  kernel_size=(3,3),  padding = 'same', input_shape=(32,32,3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64,  kernel_size=(3,3),  padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),  padding = 'same'))
model.add(layers.Dropout(0.25))

model.add(layers.Conv2D(128,  kernel_size=(3,3),  padding = 'same', input_shape=(32,32,3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128,  kernel_size=(3,3),  padding = 'same'))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2),  padding = 'same'))
model.add(layers.Dropout(0.25))

model.add(layers.Flatten())
model.add(layers.Dense(512,   activation='relu' ))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(128,   activation='relu' ))
model.add(layers.Dense(10,   activation='softmax' ))


model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics= ['accuracy'])
print (model.summary())
history = model.fit(x_train, y_train, batch_size=256, epochs=10, verbose = 1 , validation_data = (x_test,y_test))

score = model.evaluate(x_test, y_test, verbose= 1)
print(score)
