#
# come funziona un perceptrone
#
import numpy as np
input_dati = np.asarray([5,7,-3])
pesi  = np.asarray([0.3,-1,0.5])

def  perceptron(input_dati, pesi):
     return (input_dati*pesi)

risultato = perceptron(input_dati, pesi)

print (risultato)


for  r in risultato:
    if r < 0:
        print ('0')
    else:
        print ('1')

