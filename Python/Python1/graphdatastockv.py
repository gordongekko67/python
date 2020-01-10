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

def graph(s, start, end):
    # lettura dataframe
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
        # preparazione campi a video
        s = self.lineEdit.text()
        start = self.dateEdit1.text()
        end   = self.dateEdit2.text()
        # lancio grafico
        graph(s, start, end)

app = QApplication(sys.argv)
widget = Life2coding()
widget.show()
sys.exit(app.exec())



