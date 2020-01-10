import sys

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import  QApplication, QDialog, QLineEdit
from PyQt5.uic import loadUi


class Life2coding (QDialog):
    def __init__(self):
        super(Life2coding, self).__init__()
        loadUi('life2coding.ui', self)
        self.setWindowTitle('Life2coding  PyQt5 GUI')
        self.pushButton.clicked.connect(self.pushButtonc)
    @pyqtSlot()
    def pushButtonc(self):
        self.label1.setText('Welcome')


app = QApplication(sys.argv)
widget = Life2coding()
widget.show()
sys.exit(app.exec())



