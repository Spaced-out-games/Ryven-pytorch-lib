from ryven.NWENV import *

from qtpy.QtGui import QFont
from qtpy.QtCore import Qt, Signal, QEvent
from qtpy.QtWidgets import QPushButton,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout

class GatherWidget(QWidget,MWB):
	def __init__(self, params):
		MWB.__init__(self, params)
		QWidget.__init__(self)
		self.IncreaseArgsButton = QPushButton("+")
		self.IncreaseArgsButton.connect(self.node.test)
		
		self.decreaseArgsButton = QPushButton("-")
		self.layout = QVBoxLayout(self)
		self.layout.addWidget(self.IncreaseArgsButton)
		self.layout.addWidget(self.IncreaseArgsButton)



export_widgets(
	GatherWidget
)