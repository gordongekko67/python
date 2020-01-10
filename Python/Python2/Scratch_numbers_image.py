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
print(x_train.shape)
print(y_train.shape)




plt.figure(figsize = ( 16, 16))

i = 1
num = 1
for  e in eti :
      image = x_train[y_train.tolist().index(e)]
      plt.subplot(10, 10, i)
      plt.axis('off')
      #plt.title()
      i +=1
      _=plt.imshow(image, cmap = 'gray')
      print (num , x_train[y_train.tolist().index(e)])
      num +=1
plt.show()

i = 1


image = x_train[y_train.tolist().index(3)]
plt.subplot(10, 10, i)
plt.axis('off')
#plt.title()
i +=1
_=plt.imshow(image, cmap = 'inferno')
plt.show()


print ('uno')
print (x_train[y_train.tolist().index(1)])


print ('due')
print (x_train[y_train.tolist().index(2)])
