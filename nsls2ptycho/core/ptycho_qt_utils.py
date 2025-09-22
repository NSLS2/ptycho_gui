from PyQt5 import QtCore, QtGui

class PtychoStream(QtCore.QObject):
    message = QtCore.pyqtSignal(str, QtGui.QColor)
    def __init__(self, parent=None, color="black"):
        super().__init__(parent)
        self.color = QtGui.QColor(color) # output text color

    def write(self, message):
        self.message.emit(str(message), self.color)

    def flush(self):
        pass

