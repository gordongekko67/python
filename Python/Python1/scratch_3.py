from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.animation  as animation
from matplotlib import style
import datetime
import numpy
from pandas_datareader import data as web

start = datetime.datetime(2017,  1,  1)
end   = datetime.datetime(2018, 12, 31)

style.use('fivethirtyeight')

window = Tk()

#name window
window.title('My Graph ')

#window sized
window.geometry('500x600')


def graph():
    s = entry3.get()
    df = web.DataReader(s , "yahoo", start, end)
    print(df.head())
    df['Adj Close'].plot()
    plt.show()

L1 = Label(window, text='Initial data ')
entry1 = Entry(window, width=20)
L1.pack()
entry1.pack()

L2 = Label(window, text='Final date ')
entry2 = Entry(window, width=20)
L2.pack()
entry2.pack()

L3 = Label(window, text='Ticker ')
entry3 = Entry(window, width=8)
L3.pack()
entry3.pack()

button = Button(window, text='Click to Graph', command=graph)
button.pack()

window.mainloop()

