import sys
from PyQt5 import QtWidgets
from .ui import ui_reconstep
from .core.ptycho import utils

import numpy as np


class ReconStepWindow(QtWidgets.QMainWindow, ui_reconstep.Ui_MainWindow):
    def __init__(self, obj_num=1, prb_num=1, result_type_num=1, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')

        # connect
        self.slider_iters.valueChanged.connect(self.slider_iters_sb_iter_op)
        self.sb_iter.valueChanged.connect(self.slider_iters_sb_iter_op)
        self.cb_image_object.currentIndexChanged.connect(self.cb_image_object_op)
        self.cb_image_probe.currentIndexChanged.connect(self.cb_image_probe_op)
        self.btn_close.clicked.connect(self.btn_close_op)

        # TODO: enable (and rename) these buttons when they are implemented
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        #self. ...

        self.move(760, 0)

        self.reset_window(obj_num, prb_num, result_type_num)

    def reset_window(self, obj_num=1, prb_num=1, result_type_num=1, iterations=50, slider_interval=1):
        """Called from outside"""
        self.image_buffer = {}
        self.metric_buffer_it = []
        self.metric_buffer = []
        self.current_max_iters = 1
        self.progressBar.setValue(0)
        self.reset_iter(iterations, slider_interval)
        # can we reset the figures here???
        self.canvas_object_amp.reset()
        self.canvas_object_pha.reset()
        self.canvas_probe_amp.reset()
        self.canvas_probe_pha.reset()
        self.canvas_probe_fft.reset()
        self.canvas_probe_comp.reset()
        self.canvas_object_chi.reset()
        self.canvas_object_chi.axis_on()
        self.canvas_probe_chi.reset()
        self.canvas_probe_chi.axis_on()
        self.reset_image_menu(obj_num, prb_num, result_type_num)

    def reset_image_menu(self, obj_num, prb_num, result_type_num):
        # number of object and probe images
        self.obj_num = obj_num
        self.prb_num = prb_num

        # number of result types
        # mode: 1 (orth_ave_rp); multislice: 1 (ave_rp); otherwise: 2 (ave & ave_rp)
        self.result_type_num = result_type_num

        self.cb_image_object.clear()
        for i in range(self.obj_num):
            self.cb_image_object.addItem("Object " + str(i))

        self.cb_image_probe.clear()
        for i in range(self.prb_num):
            self.cb_image_probe.addItem("Probe " + str(i))

    def is_live_update(self):
        return self.ck_live.isChecked()

    def slider_iters_sb_iter_op(self, it):
        # lock the slider position to available values
        temp = (it - self.sb_iter.minimum()) % self.sb_iter.singleStep()
        if temp == 0:
            pass
        elif temp >= self.sb_iter.singleStep() - temp:
            it += self.sb_iter.singleStep() - temp
        else:
            it -= temp
            
        self.slider_iters.setValue(it)
        self.sb_iter.setValue(it)
        self.update_images(it)

    def cb_image_object_op(self, idx):
        it = self.sb_iter.value()
        if it in self.image_buffer:
            images_to_show = self.image_buffer[it]
            object_image = self._fetch_images(it, images_to_show, 'obj_amp')
            if object_image is not None:
                self.canvas_object_amp.update_image(object_image)
            object_image = self._fetch_images(it, images_to_show, 'obj_pha')
            if object_image is not None:
                self.canvas_object_pha.update_image(object_image)

    def cb_image_probe_op(self, idx):
        it = self.sb_iter.value()
        if it not in self.image_buffer:
            it = len(self.image_buffer)-1
        if it in self.image_buffer:
            images_to_show = self.image_buffer[it]
            probe_image_amp = self._fetch_images(it, images_to_show, 'prb_amp')
            if probe_image_amp is not None:
                self.canvas_probe_amp.update_image(probe_image_amp)
            probe_image_pha = self._fetch_images(it, images_to_show, 'prb_pha')
            if probe_image_pha is not None:
                self.canvas_probe_pha.update_image(probe_image_pha)
            if probe_image_amp is not None and probe_image_pha is not None:
                probe_image_comp = utils.imRGB_from_comp(probe_image_amp,probe_image_pha)
                self.canvas_probe_comp.update_image(probe_image_comp)
                
                probe_comp = probe_image_amp*np.exp(1j*probe_image_pha)
                probe_fft = np.abs(np.fft.fftshift(np.fft.fft2(probe_comp)))
                self.canvas_probe_fft.update_image(probe_fft)

    def btn_close_op(self):
        # todo: close with signal to main window
        self.reset_window()
        self.close()

    def reset_iter(self, max_iters, interval):
        self.progressBar.setValue(0)
        self.progressBar.setMaximum(max_iters)

        # one-based index
        self.max_iters = max_iters
        self.current_max_iters = 1
        self.slider_iters.setValue(1)
        self.slider_iters.setMinimum(1)
        self.slider_iters.setMaximum(1)
        self.slider_iters.setSingleStep(interval)
        self.sb_iter.setValue(1)
        self.sb_iter.setMinimum(1)
        self.sb_iter.setMaximum(1)
        self.sb_iter.setSingleStep(interval)

    def update_iter(self, it):
        """Called from outside"""
        self.progressBar.setValue(it)

        if self.current_max_iters < it:
            self.slider_iters.setMaximum(it)
            self.sb_iter.setMaximum(it)
            self.current_max_iters = it

        if self.is_live_update():
            self.slider_iters.setValue(it)
            self.sb_iter.setValue(it)

    def update_images(self, it, images=None):
        if images is not None:
            # just hold the mmap reference, don't do expansive copy
            self.image_buffer[it] = images

        images_to_show = None
        if self.is_live_update() and images is not None:
            images_to_show = images
        elif it in self.image_buffer:
            images_to_show = self.image_buffer[it]

        object_image_amp = None
        probe_image_amp = None
        object_image_pha = None
        probe_image_pha = None

        if not hasattr(self,'it_ondisplay'):
            self.it_ondisplay = -1

        if images_to_show is not None and it != self.it_ondisplay:
            object_image_amp = self._fetch_images(it, images_to_show, 'obj_amp')
            probe_image_amp = self._fetch_images(it, images_to_show, 'prb_amp')
            if object_image_amp is not None:
                self.canvas_object_amp.update_image(object_image_amp)
            if probe_image_amp is not None:
                self.canvas_probe_amp.update_image(probe_image_amp)

            object_image_pha = self._fetch_images(it, images_to_show, 'obj_pha')
            probe_image_pha = self._fetch_images(it, images_to_show, 'prb_pha')
            if object_image_pha is not None:
                self.canvas_object_pha.update_image(object_image_pha)
            if probe_image_pha is not None:
                self.canvas_probe_pha.update_image(probe_image_pha)

            #if object_image_amp is not None and object_image_pha is not None:
            #    object_image_comp = utils.imRGB_from_comp(object_image_amp,object_image_pha,(0.95,0.05))
            #    self.canvas_object_comp.update_image(object_image_comp)
            
            if probe_image_amp is not None and probe_image_pha is not None:
                probe_image_comp = utils.imRGB_from_comp(probe_image_amp,probe_image_pha)
                self.canvas_probe_comp.update_image(probe_image_comp)

                probe_comp = probe_image_amp*np.exp(1j*probe_image_pha)
                probe_fft = np.abs(np.fft.fftshift(np.fft.fft2(probe_comp)))
                self.canvas_probe_fft.update_image(probe_fft)
            
            self.it_ondisplay = it

            

            



    def _fetch_images(self, it, images_to_show, flag=None):
        image = None
        if flag.startswith('prb'):
            prb_idx = self.cb_image_probe.currentIndex()
            offset = 0 if flag.endswith('amp') else 1
            if it <= self.current_max_iters and prb_idx < self.prb_num:
                # the factor of 2 below is due to two kind of functions being applied to obj (angle and abs)
                image = images_to_show[2*(prb_idx+self.obj_num)+offset]
            elif it > self.current_max_iters:
                # ptycho completes, perform post processing
                # the factor of 2 below is due to two kind of functions being applied to obj (angle and abs)
                image = images_to_show[2*(prb_idx+self.result_type_num*self.obj_num)+offset]
        elif flag.startswith('obj'):
            obj_idx = self.cb_image_object.currentIndex()
            offset = 0 if flag.endswith('pha') else 1
            if it <= self.current_max_iters and obj_idx < self.obj_num:
                # the factor of 2 below is due to two kind of functions being applied to obj (angle and abs)
                image = images_to_show[2*obj_idx+offset]
            elif it > self.current_max_iters:
                # ptycho completes, perform post processing
                # the factor of 2 below is due to two kind of functions being applied to obj (angle and abs)
                image = images_to_show[2*(obj_idx)+offset]
        else:
            # shouldn't happen!
            pass

        return image

    def update_metric(self, it, data):
        self.metric_buffer_it.append(it)
        self.metric_buffer.append(data)

        # get key from combo box
        object_to_plot = np.array([ d['object_chi'] for d in self.metric_buffer])
        probe_to_plot  = np.array([ d['probe_chi'] for d in self.metric_buffer])
        self.canvas_object_chi.update_plot(self.metric_buffer_it, object_to_plot)
        self.canvas_probe_chi.update_plot(self.metric_buffer_it, probe_to_plot)

    def debug(self):
        ''' called from MainWindow'''
        for key in self.image_buffer:
            print("#{}: ".format(key), end='', file=sys.stderr)
            for item in self.image_buffer[key]:
                print("{} ".format(hex(id(item))), end='', file=sys.stderr)
            print("", file=sys.stderr)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    w = ReconStepWindow()
    w.show()

    sys.exit(app.exec_())
