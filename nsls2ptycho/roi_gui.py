import sys
from PyQt5 import QtWidgets
from .ui import ui_roi

import numpy as np
from .core.widgets.imgTools import find_outlier_pixels, find_brightest_pixels, rm_outlier_pixels
from .core.widgets.badpixel_dialog import BadPixelDialog
from .core.databroker_api import save_data


class RoiWindow(QtWidgets.QMainWindow, ui_roi.Ui_MainWindow):

    def __init__(self, parent=None, image=None, param=None, main_window=None, overflow_value=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')

        self.main_window = main_window # need to know the caller
        self.overflow_value = overflow_value

        #image = np.load('./34784_frame0.npy')
        if image is not None:
            roi = [main_window.sp_batch_x0.value(),main_window.sp_batch_y0.value(),main_window.sp_batch_width.value(),main_window.sp_batch_height.value()]
            self.canvas.draw_image(image, cmap='gray', init_roi=roi, use_log=False)

        # signal
        self.roi_changed = self.canvas.roi_changed
        #self.reset = self.canvas.reset

        # connect
        #self.btn_badpixels_outliers.clicked.connect(self.find_badpixels)
        self.btn_badpixels_correct.clicked.connect(self.correct_badpixels)
        self.btn_badpixels_save.clicked.connect(self.save_badpixels)
        self.btn_badpixels_load.clicked.connect(self.load_badpixels)
        self.ck_show_badpixels.clicked.connect(self.show_badpixels)
        self.btn_save_to_h5.clicked.connect(self.save_to_h5_ui)
        self.ck_logscale.toggled.connect(self.use_logscale)
        self.actionBadpixels.triggered.connect(self.open_badpixel_dialog)
        self.canvas.btn_home.clicked.connect(self.reset_window)

        # badpixels
        self.badpixel_dialog = None
        #self.badpixels_method = BADPIX_OUTLIERS # method applied to get the badpixels
        # badpixel_file = main_window.le_batch_badpixel.text()
        # if badpixel_file is not None and len(badpixel_file) > 0:
        #     with open(badpixel_file, 'r') as f:
        #         for line in f:
        #             x, y = map(int, line.strip().split())
        #             self.canvas.update_overlay((y, x),True)

        # for h5 operation
        self.roi_width = None
        self.roi_height = None
        self.cx = None
        self.cy = None
        self.sp_threshold.setValue(0.0)

    def reset_window(self):
        # When this function is called, self.canvas._on_reset() is also called

        #self.canvas.reset()
        # TODO: reset bad pixels stored in canvas
        #if image is not None:
        #    self.canvas.draw_image(image)

        #self.badpixels = None
        #self.offset_x = None
        #self.offset_y = None

        self.roi_width = None
        self.roi_height = None
        self.cx = None
        self.cy = None
        self.sp_threshold.setValue(0.0)

        #self.btn_badpixels_brightest.setChecked(False)
        #self.btn_badpixels_outliers.setChecked(False)
        self.btn_badpixels_correct.setChecked(False)
        self.ck_show_badpixels.setChecked(False)
        self.ck_logscale.setChecked(False)
       
    def open_badpixel_dialog(self):
        badpixels = self.canvas.get_badpixels()
        self.badpixel_dialog = BadPixelDialog(self, badpixels)
        #todo: signal-slot for interactive list view
        self.badpixel_dialog.show()

    def find_badpixels(self):
        # always find badpixels over original image data
        img = self.canvas.image
        if img is None:
            return

        height, width = img.shape
        roi = self.canvas.get_red_roi()
        if roi is None:
            roi_img = img
            x0 = 0
            y0 = 0
            roi_width = width
            roi_height = height
        else:
            x0 = np.clip([roi[0]], 0, width-1)[0]
            y0 = np.clip([roi[1]], 0, height-1)[0]
            roi_width = roi[2]
            roi_height = roi[3]

            region = (
                slice(y0, y0 + roi_height),
                slice(x0, x0 + roi_width)
            )
            roi_img = img[region]

        # badpixels = find_outlier_pixels(roi_img)
        # badpixels = self.canvas.get_badpixels()

        self.roi_width = roi_width
        self.roi_height = roi_height
        # TEST: ROI center
        self.cx = x0 + roi_width // 2
        self.cy = y0 + roi_height // 2

        # self.canvas.set_overlay(badpixels[0], badpixels[1])
        #todo: update badpixel_dialog if it is opened
        self.ck_show_badpixels.setChecked(True)
        self.btn_badpixels_correct.setEnabled(True)

    def correct_badpixels(self):
        if self.overflow_value is not None:
            overflow_indices = np.where(self.canvas.image==self.overflow_value)
            self.canvas.set_overlay(overflow_indices[0],overflow_indices[1])
        badpixels = self.canvas.get_badpixels()
        if badpixels is None: return
        
        img = self.canvas.image
        img = rm_outlier_pixels(img, badpixels[0], badpixels[1])

        self.canvas.draw_image(img, use_log = self.ck_logscale.checkState())
        # self.canvas.clear_overlay()
        # update badpixels with corrected one
        # self.find_badpixels()

    def show_badpixels(self, state):
        self.canvas.show_overlay(state)
        
    def save_badpixels(self):
        badpixels = self.canvas.get_badpixels()
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save bad pixels to txt', filter="(*.txt)")
        if filename is not None and len(filename) > 0:
            if filename[-4:] != ".txt":
                filename += ".txt"
            with open(filename, 'w') as f:
                if badpixels is not None:
                    for i in range(len(badpixels[0])):
                        f.write("%d %d\n"%(badpixels[0][i],badpixels[1][i]))
    
    def load_badpixels(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load bad pixels from txt (add to current selection)', filter="(*.txt)")
        if filename is not None and len(filename) > 0:
            with open(filename, 'r') as f:
                for line in f:
                    x, y = map(int, line.strip().split())
                    self.canvas.update_overlay((y, x),True)

    def use_logscale(self, state):
        self.canvas.draw_image(self.canvas.image,use_log=state)

    def save_to_h5_ui(self):
        master = self.main_window
        if master is None:
            return
    
        # get threshold
        threshold = self.sp_threshold.value()
        upsample = self.sp_upsample.value()

        # get bad pixels
        # TODO: need a better foolproof way
        # to update roi_width, roi_height, cx, cy
        # if correction is made internally, correct button should be deprecated
        # because after correction bad pixels are updated according to corrected image.
        self.find_badpixels()
        badpixels = self.canvas.get_badpixels()
        # TODO: separate this part to another function?

        if badpixels is not None and len(badpixels) == 2 and len(badpixels[0]) == len(badpixels[1]) > 0:
            pass
            print("%d bad pixels will be corrected"%len(badpixels[0]))
            #for y, x in zip(badpixels[0], badpixels[1]):
            #    print("  (x, y) = ({0}, {1})".format(x, y))
        else:
            badpixels = []
            print("no bad pixels")
    
        # get blue rois
        blue_rois = self.canvas.get_blue_roi()
        #print(blue_rois)

        master.save_to_h5(self.roi_width, self.roi_height, self.cx, self.cy, threshold, badpixels, blue_rois, upsample, save_diff = False)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    w = RoiWindow()
    w.show()

    sys.exit(app.exec_())
