# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nsls2ptycho/ui/ui_scan.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(490, 360)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMinimumSize(QtCore.QSize(360, 360))
        self.centralwidget.setObjectName("centralwidget")
        self.scatter_pt = MplCanvas(self.centralwidget)
        self.scatter_pt.setGeometry(QtCore.QRect(5, 5, 480, 350))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.scatter_pt.sizePolicy().hasHeightForWidth())
        self.scatter_pt.setSizePolicy(sizePolicy)
        self.scatter_pt.setMinimumSize(QtCore.QSize(0, 0))
        self.scatter_pt.setObjectName("scatter_pt")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "scan points"))

from ..core.widgets.mplcanvas import MplCanvas
