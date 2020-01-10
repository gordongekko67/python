import numpy as np
from keras.utils  import np_utils

import keras
import os
from keras.models import Sequential
from keras import layers
from keras.optimizers  import SGD

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# lettura dei due files
#df = pd.read_csv("/Users/Enrico/Downloads/LANL-Earthquake-Prediction/train.csv")



import os
path = '/Users/Enrico/Downloads/LANL-Earthquake-Prediction/test'
folder = os.fsencode(path)
filenames = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.csv') ): # whatever file types you're using...
        filenames.append(filename)

filenames.sort() # now you have the filenames and can do something with them
print (filenames)

for file in filenames:
       stringa1 = "/Users/Enrico/Downloads/LANL-Earthquake-Prediction/"
       stringa2 = str(filenames[file])
       stringa3 = ".csv"
       stringa  = stringa1 + stringa2 + stringa3
       #dl = pd.read_csv(stringa)
       print (stringa)
