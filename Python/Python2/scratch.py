import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras as ks
import pandas as pd

from  keras.layers import Dense
from  keras.models import Sequential
from  keras.optimizers import adam

#df = pd.read_csv('https://archive.ics.uci.edu/ml/datasets/Iris/Iris.data', header = 0)
#print (df)



np.random.seed(12345)
x = np.random.standard_normal(100)
y = 2 * x *10
print(x)
print(y)
plt.plot(x,y , "ro")
plt.show()


#df = pd.read_csv('http://archive.ics.uci.edu/ml/datasets/Iris', header = 0)
#print (df)






