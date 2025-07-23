from PyQt5 import QtCore, QtGui, QtWidgets
from ..utils import parse_range2
from ...ui import ui_list_dialog


#def _dragEnterEvent(listWidget, event):
#    #event.setDropAction(QtCore.Qt.MoveAction)
#    event.accept()
#
#def _dragMoveEvent(listWidget, event):
#    #event.setDropAction(QtCore.Qt.MoveAction)
#    event.accept()
#
#def _dropEvent(listWidget, event):
#    #event.setDropAction(QtCore.Qt.CopyAction)
#    event.accept()
#    listWidget.addItem(str(event.mimeData().text()))


class ListWidget(QtWidgets.QDialog, ui_list_dialog.Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')
        self.setObjectName("Assoc.Scans")

        self.listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        #self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.listWidget.setAcceptDrops(True)
        self.listWidget.setDragEnabled(True)
        self.listWidget.setDropIndicatorShown(True)
        self.listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        #self.listWidget.dragEnterEvent = lambda event: _dragEnterEvent(self.listWidget, event)
        #self.listWidget.dragMoveEvent = lambda event: _dragMoveEvent(self.listWidget, event)
        #self.listWidget.dropEvent = lambda event: _dropEvent(self.listWidget, event)

        self.btn_add_item.clicked.connect(self.add_item)
        self.btn_rm_item.clicked.connect(self.remove_item)
#        self.btn_close.clicked.connect(self.close_dialog)

        self.le_input.setToolTip(QtCore.QCoreApplication.translate("Assoc.Scans", "Set scan numbers and ranges. Example: 128-131, 137-139"))

#    def close_dialog(self):
#        self.destroy()

    def add_item(self):
        items = parse_range2(self.le_input.text(), batch_processing=False)
        self.listWidget.addItems([str(item) for item in items])
        self.le_input.setText('')
        #self.listWidget.sortItems()

    def remove_item(self):
        item = self.listWidget.takeItem(self.listWidget.currentRow())
        #self.listWidget.sortItems()
        del item
