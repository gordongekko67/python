from numpy import exp, array, random, dot

train = array([[0,0,1], [1,1,1,], [1,0,1],[0,1,1]])
cl = array ([[0,1,1,0]]).T

print (train)
print (cl)

random.seed(12345)
pesi = 2*random.random((3,1)) -1
print (pesi)

random.seed(12345)
for iteration in range(10000):
    #output = 1/(  1+ exp-(  dot(train, pesi()))            )
    #pesi += dot()



