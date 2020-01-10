import numpy as np
import matplotlib.pyplot as plt


from  keras.layers import Dense
from  keras.models import Sequential
from  keras.optimizers import Adam

# imporsto un seed
np.random.random(12345)

x = np.random.standard_normal(100)
y = 2*x *10

#print (x, y)

plt.plot(x, y, 'ro')
#plt.show()

train = x [:80]
test  = x [81:]

cl_train = y [:80]
cl_test  = y [81:]

#inizializzo il modello
model = Sequential()
# aggiungo i vari strati
model.add(Dense(1, input_shape=(1, ), activation=None))

p = model.summary()
print (p)

model.compile(loss='mse', optimizer = 'Adam' , metrics=['mae'])
history = model.fit(train, cl_train, batch_size=128, epochs=300, verbose = 1, validation_data=(train, cl_train))


