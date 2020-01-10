#  questo e' python
print ("w la libertà")
print (3+2)
print ("2 + 2 is",  print (2+2))
print ("Enrico ")
print ("Saccheggiani")
#print "3 * 4 is", 3 * 4
#print 100 - 1, " = 100 - 1"
#print "(33 + 2) / 5 + 11.5 = ",(33 + 2) / 5 + 11.5


print ("Halt!")
s = input("Who Goes there? ")
print ("You may pass,", s)


n = int(input('Inserisci un numero: '))
if n < 0:
    print(n, 'è negativo')
elif n > 0:
    print(n, 'è positivo')
elif n < 6:
    print(n, 'è minore di 6')
else:
    print(n, 'è zero')

# if
n1 = input("Number? ")
n = int(n1)
if n < 0:
        print ("The absolute value of",n,"is",-n)
else:
        print ("The absolute value of",n,"is",n)


str1 = input("Stringa1")
str2 = input("stringa2")
str3 = str1+str2
print (str3)
num1 = input ("inserisci il primo numero")
n1 = int(num1)
num2 = input ("inserisci il secondo  numero")
n2 = int(num2)

print(n1*n2)



# substring
s = 'Python'
a = s[0:2]

print (a)
n = int(input('Inserisci un numero: '))
if n < 0:
    print(n, 'è negativo')
elif n > 0:
    print(n, 'è positivo')
else:
    print(n, 'è zero')


n = int(input('Inserisci un numero : '))
if n == 0:
    # se il numero è zero
    print(n, 'è zero')
else:
    # se il numero non è zero
    if n > 0:
        print(n, 'è positivo')
    else:
        print(n, 'è negativo')

 # esempio in Python  switch
if n == 0:
    print('zero')
elif n == 1 or n == 2:
    print('uno o due')
elif n == 3:
    print('tre')
else:
    print('numero diverso da 0, 1, 2, 3')

##########################################################################################################
#  ciclo for
##########################################################################################################
# stampa il quadrato di ogni numero di seq
seq = [1, 2, 3, 4, 5]
for n in seq:
    print('Il quadrato di', n, 'è', n**2)


# for in seq
seq = [1, 2, 3, 4, 5]
for n in seq:
        print('Il numero', n, 'è', end=' ')
        if n%2 == 0:
           print('pari')
        else:
           print('dispari')



# for in range
for n in range(1, 6):
       print('Il quadrato in range ', n, 'è', n**2)

# Measure some strings:
words = ['cat', 'window', 'defenestrate']
for w in words:
    print(w, len(w))

#################################################################################################
# ciclo
#################################################################################################
a = 0
while a < 100:
        a = a + 1
        print (a)

# while
a = 0
while a < 10:
        a = a + 1
        if a > 5:
                print (a," > ",5)
        elif a <= 7:
                print (a," <= ",7)
        else:
                print ("Neither test was true")

###########################################################################################
#      funzioni
############################################################################################

# funzione senza ritorno parametri
def say_hello(name):
     print('Hello {}!'.format(name))


say_hello("cicciuzzo")

#funzione con ritorno parametri
def square(n):
      return n**2
x = square(5)
print (x)



###########################################################################################
# modello try catch
###########################################################################################
try:
  f = open("demofile.txt")
  f.write("Lorum Ipsum")
except:
  print("Something went wrong when writing to the file")
finally:
  f.close()

  try:
      something()
  except Exception as e:
      send_somewhere(traceback.format_exception(*sys.exc_info()))
      raise  # reraises the exception

###################################################################################################
#   multithreading
###################################################################################################

import time, threading

#funzione thread

def show():
    print("start thread")
    for i in range(5):
        print(i,time.time())
        time.sleep(1.5)
    print("fine thread")

# programma principale
print("Inizio main")
t= threading.Thread(target=show)
t.start()
time.sleep(10)
print("fine programma")