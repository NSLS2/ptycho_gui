import sys

import numpy as np
from PyQt5 import QtWidgets

from .ui import ui_scan

class ScanWindow(QtWidgets.QMainWindow, ui_scan.Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')
        self.move(760+600, 0)
        self.reset_window()

    def reset_window(self):
        """Called from outside"""
        # can we reset the figures here???
        self.scatter_pt.reset()

    def update_image(self, data, num_process):
        self.scatter_pt.reset()
        self.scatter_pt.update_scatter(data, num_process)

    def debug(self):
        ''' called from MainWindow'''
        for key in self.image_buffer:
            print("#{}: ".format(key), end='', file=sys.stderr)
            for item in self.image_buffer[key]:
                print("{} ".format(hex(id(item))), end='', file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    w = ScanWindow()
    w.show()

    n_processes = 4
    n_pts = 500
    data = np.random.random((2, n_pts))
    w.update_image(data, n_processes)
    data = np.random.random((2, n_pts))
    w.update_image(data, n_processes)

    sys.exit(app.exec_())
