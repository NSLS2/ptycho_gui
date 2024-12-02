from PyQt5 import QtCore, QtGui, QtWidgets
from ...ui import ui_badpixels

# todo: interactive item add and delete
class BadPixelDialog(QtWidgets.QDialog, ui_badpixels.Ui_Dialog):
    def __init__(self, parent=None, badpixels=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')

        if badpixels is not None:
            items = ["{:5d}, {:5d}".format(col, row) for row, col in zip(badpixels[0], badpixels[1])]
            self.listWidget.addItems(items)
            self.listWidget.sortItems()


        self.btn_close.clicked.connect(self.close_dialog)

    def close_dialog(self):
        self.destroy()
