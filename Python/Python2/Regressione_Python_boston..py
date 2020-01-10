from keras import layers
from keras import models
from keras import optimizers
from keras import losses
from keras  import metrics

from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print (train_data.shape)
print (test_data.shape)

print (train_targets.shape, test_targets.shape)

mean = train_data.mean(axis = 0)
std  = train_data.std(axis = 0)
train_data-=mean
train_data /= std

test_data -= mean
test_data -= std

model = models.Sequential()
model.add(layers.Dense(64,  activation='relu', input_shape = (train_data.shape[1],)))
model.add(layers.Dense(64,  activation='relu'))
model.add(layers.Dense(1))

model.compile( optimizer='rmsprop', loss='mse', metrics= ['mae'])
# adattiamo modello ai dati
model.fit(train_data, train_targets, epochs= 100, batch_size= 1, verbose= 1)

mse, mae = model.evaluate(test_data, test_targets, verbose = 0 )
print (mse)
print (mae)

