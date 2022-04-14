from ryven.NWENV import *
from qtpy.QtWidgets import QWidget

class SomeMainWidget(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)
    ...

class SomeInputWidget(IWB, QWidget):
    def __init__(self, params):
        IWB.__init__(self, params)
        QWidget.__init__(self)
    ...

export_widgets(
    SomeMainWidget,
    SomeInputWidget,
)