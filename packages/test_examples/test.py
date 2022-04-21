'''--------------------------------------------------------------IMPORTS---------------------------------------------------------------'''
import sys
import random
from PySide6 import QtCore, QtWidgets, QtGui
'''--------------------------------------------------------------IMPORTS---------------------------------------------------------------'''



'''--------------------------------------------------------------WIDGETS---------------------------------------------------------------'''
class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.button = QtWidgets.QPushButton("Click me!")
        self.text = QtWidgets.QLabel("Hello World",
                                     alignment=QtCore.Qt.AlignCenter)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button)
'''--------------------------------------------------------------WIDGETS---------------------------------------------------------------'''



'''----------------------------------------------------------------APP-----------------------------------------------------------------'''
if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = MyWidget()
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec())
'''----------------------------------------------------------------APP-----------------------------------------------------------------'''
