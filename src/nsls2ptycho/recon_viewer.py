import sys,time,os
from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget

import numpy as np
import traceback

from .reconStep_gui import ReconStepWindow

run = True

def update_window(w,live_path):
    os.listdir(live_path)
    prb_live_file = os.path.join(live_path,'prb_live.npy')
    obj_live_file = os.path.join(live_path,'obj_live.npy')
    if os.path.exists(prb_live_file) and os.path.getsize(prb_live_file)>0 and os.path.exists(obj_live_file) and os.path.getsize(obj_live_file)>0:
    #time.sleep(1) # wait for the npy files in file system
        try:
            _prb_live = np.load(prb_live_file)
            _obj_live = np.load(obj_live_file)

            # Temporary
            _obj_live = np.flip(_obj_live,1)

            if np.sum(np.abs(_prb_live))>0 and np.sum(np.abs(_obj_live))>0:
                images = []
                images.append(np.rot90(np.angle(_obj_live[0])))
                images.append(np.rot90(np.abs(_obj_live[0])))
                images.append(np.rot90(np.abs(_prb_live[0])))
                images.append(np.rot90(np.angle(_prb_live[0])))
                w.it_ondisplay = -1
                w.update_images(0,images)
        except:
            pass
    QTimer.singleShot(1000, lambda: update_window(w,live_path))

def scale_window(window, scale_factor):
    geometry = window.geometry()
    window.setGeometry(geometry.x(), geometry.y(),
                       int(geometry.width() * scale_factor),
                       int(geometry.height() * scale_factor * 0.75))
    window.centralWidget().setFixedSize(int(geometry.width()),int(geometry.height()*0.75))

    for widget in window.findChildren(QWidget):
        widget.setFixedSize(int(widget.width() * scale_factor), int(widget.height() * scale_factor))
        widget.move(int(widget.x() * scale_factor), int(widget.y() * scale_factor))

def main():
    app = QtWidgets.QApplication(sys.argv)

    if len(sys.argv) > 1:
        scale_factor = float(sys.argv[1])
    else:
        scale_factor = 2.5

    if len(sys.argv) > 2:
        live_path = sys.argv[2]
    else:
        live_path = '/nsls2/data/hxn/legacy/users/Holoscan'


    w = ReconStepWindow()

    scale_window(w,scale_factor)
    update_window(w,live_path)
    w.show()

    w.closeEvent = lambda event: event.accept()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()