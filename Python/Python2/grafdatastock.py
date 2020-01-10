import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import  QApplication, QDialog
from PyQt5.uic import loadUi

import matplotlib.pyplot as plt
import matplotlib.animation  as animation
from matplotlib import style
import datetime
import numpy
from pandas_datareader import data as web
s = ""

start = datetime.datetime(2017,  1,  1)
end   = datetime.datetime(2018, 12, 31)



def graph(s, start, end):
    print(s)
    df = web.DataReader(s , "yahoo", start, end)
    print(df.head())
    df['Adj Close'].plot()
    plt.show()

class Life2coding (QDialog):
    def __init__(self):
        super(Life2coding, self).__init__()
        loadUi('graphdata.ui', self)
        self.setWindowTitle('Graphdata  PyQt5 GUI')
        self.pushButton.clicked.connect(self.pushButtonc)
    @pyqtSlot()
    def pushButtonc(self):
        s = self.lineEdit.text()
        graph(s, start, end)

app = QApplication(sys.argv)
widget = Life2coding()
widget.show()
sys.exit(app.exec())



