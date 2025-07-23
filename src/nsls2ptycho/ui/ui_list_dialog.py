# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'nsls2ptycho/ui/ui_list_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(316, 271)
        self.le_input = QtWidgets.QLineEdit(Form)
        self.le_input.setGeometry(QtCore.QRect(185, 70, 113, 21))
        self.le_input.setObjectName("le_input")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(180, 10, 121, 51))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(180, 160, 121, 51))
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")
        self.listWidget = QtWidgets.QListWidget(Form)
        self.listWidget.setGeometry(QtCore.QRect(21, 21, 141, 231))
        self.listWidget.setObjectName("listWidget")
        self.btn_add_item = QtWidgets.QPushButton(Form)
        self.btn_add_item.setGeometry(QtCore.QRect(200, 100, 80, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_add_item.sizePolicy().hasHeightForWidth())
        self.btn_add_item.setSizePolicy(sizePolicy)
        self.btn_add_item.setMinimumSize(QtCore.QSize(80, 0))
        self.btn_add_item.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_add_item.setObjectName("btn_add_item")
        self.btn_rm_item = QtWidgets.QPushButton(Form)
        self.btn_rm_item.setGeometry(QtCore.QRect(200, 220, 80, 32))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_rm_item.sizePolicy().hasHeightForWidth())
        self.btn_rm_item.setSizePolicy(sizePolicy)
        self.btn_rm_item.setMinimumSize(QtCore.QSize(80, 0))
        self.btn_rm_item.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_rm_item.setObjectName("btn_rm_item")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Input here and click Add:"))
        self.label_2.setText(_translate("Form", "Or, select item from the left and click Remove:"))
        self.btn_add_item.setText(_translate("Form", "Add"))
        self.btn_rm_item.setText(_translate("Form", "Remove"))

