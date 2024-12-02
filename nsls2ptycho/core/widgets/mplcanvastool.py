import os
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
import numpy as np

from .imgTools import estimate_roi
from .eventhandler import EventHandler


class MplCanvasTool(QtWidgets.QWidget):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        super().__init__(parent)

        fig = Figure(figsize=(width, height), dpi=dpi)
        ax = Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        self.ax = ax
        self.fig = fig
        self.canvas = FigureCanvas(fig)

        # initialized by _get_roi_bar()
        self.sp_x0 = None
        self.sp_y0 = None
        self.sp_w = None
        self.sp_h = None
        self._roi_all = None
        self.ref_roi_side = [64, 96, 128, 160, 192, 224, 256] # x 32, square

        self._actions = {}
        self._active = None
        self._eventHandler = EventHandler()
        self.roi_changed = self._eventHandler.roi_changed
        self._ids = []

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(self._get_toolbar())
        layout.addLayout(self._get_roi_bar())
        layout.addLayout(self._get_help_bar())
        self.setLayout(layout)

        self._update_help(0)

        self._eventHandler.brush_changed.connect(self.update_overlay)

        self.image = None
        self.image_data = None
        self.image_handler = None
        self.overlay = None
        self.overlay_handler = None
        self.reset()

    def _get_toolbar(self):
        self.btn_home = QtWidgets.QPushButton('RESET')
        #self.btn_pan_zoom = QtWidgets.QPushButton('PAN/ZOOM')
        self.btn_roi = QtWidgets.QPushButton('ROI')
        self.btn_brush = QtWidgets.QPushButton('BRUSH')
        #self.btn_roi_adjust = QtWidgets.QPushButton('ADJUST')

        #self.btn_pan_zoom.setCheckable(True)
        self.btn_roi.setCheckable(True)
        self.btn_brush.setCheckable(True)

        self.btn_home.clicked.connect(self._on_reset)
        #self.btn_pan_zoom.clicked.connect(lambda: self._update_buttons('pan/zoom'))
        self.btn_roi.clicked.connect(lambda: self._update_buttons('roi'))
        self.btn_brush.clicked.connect(lambda: self._update_buttons('brush'))
        #self.btn_roi_adjust.clicked.connect(self._on_adjust_roi)

        #self._actions['pan/zoom'] = self.btn_pan_zoom
        self._actions['roi'] = self.btn_roi
        self._actions['brush'] = self.btn_brush

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.btn_home)
        #layout.addWidget(self.btn_pan_zoom)
        layout.addWidget(self.btn_roi)
        #layout.addWidget(self.btn_roi_adjust)
        layout.addWidget(self.btn_brush)
        return layout

    def _get_roi_bar(self):

        self.sp_x0 = QtWidgets.QSpinBox(self)
        self.sp_y0 = QtWidgets.QSpinBox(self)
        self.sp_w = QtWidgets.QSpinBox(self)
        self.sp_h = QtWidgets.QSpinBox(self)
        self._roi_all = [self.sp_x0, self.sp_y0, self.sp_w, self.sp_h]
        for sp in self._roi_all:
            sp.setMaximum(9999)
            sp.setMinimum(0)
            sp.setValue(0)
            sp.valueChanged.connect(self._update_roi_canvas)

        self.coord_label = QtWidgets.QLabel('(x, y), value')

        self._eventHandler.roi_changed.connect(self._update_roi)
        self._eventHandler.coord_changed.connect(self._update_coord)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel('x0'))
        layout.addWidget(self.sp_x0)
        layout.addWidget(QtWidgets.QLabel('y0'))
        layout.addWidget(self.sp_y0)
        layout.addWidget(QtWidgets.QLabel('w'))
        layout.addWidget(self.sp_w)
        layout.addWidget(QtWidgets.QLabel('h'))
        layout.addWidget(self.sp_h)
        layout.addWidget(self.coord_label)
        spacerItem = QtWidgets.QSpacerItem(0,0,QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Preferred)
        layout.addItem(spacerItem)
        return layout
    
    def _get_help_bar(self):
        self.help_bar = QtWidgets.QLabel('Help')
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.help_bar)
        return layout
    
    def _update_help(self,index = 0):
        messages = {
            0: "LMB to pan image, scroll to zoom.",
            1: "LMB click to select ROI, LMB drag to move or create red ROI (data),\nRMB drag to create blue ROI (removal), LMB double click to remove ROI",
            2: "LMB click or drag to mark pixels to remove, RMB to unmark."
        }
        self.help_bar.setText(messages.get(index,"Invalid index"))

    def _update_coord(self, ix, iy):

        if self.image is None or ix < 0 or ix >= self.image.shape[1] or iy < 0 or iy >= self.image.shape[0]:
            value = 'None'
        else:
            value = '{:.2e}'.format(self.image[iy, ix])

        self.coord_label.setText('({:d}, {:d}), {:s}'.format(ix, iy, value))

    def _update_roi(self, x0, y0, w, h):
        for sp in self._roi_all: sp.valueChanged.disconnect(self._update_roi_canvas)

        self.sp_x0.setValue(x0)
        self.sp_y0.setValue(y0)
        self.sp_w.setValue(w)
        self.sp_h.setValue(h)

        for sp in self._roi_all: sp.valueChanged.connect(self._update_roi_canvas)

    def _update_roi_canvas(self):
        x0 = self.sp_x0.value()
        y0 = self.sp_y0.value()
        w = self.sp_w.value()
        h = self.sp_h.value()
        if w <= 0 or h <= 0: return
        self._eventHandler.set_curr_roi(self.ax, (x0, y0), w, h)

    def _update_buttons(self, op_name):
        if self._active == op_name:
            self._active = None
        else:
            self._active = op_name

        #self._actions['pan/zoom'].setChecked(self._active == 'pan/zoom')
        self._actions['roi'].setChecked(self._active == 'roi')
        self._actions['brush'].setChecked(self._active == 'brush')

        for id in self._ids:
            self.canvas.mpl_disconnect(id)
        self._ids = []

        #if self._active == 'pan/zoom':
        #    self._ids = self._eventHandler.zoom_pan_factory(self.ax)
        #el
        if self._active == 'roi':
            self._ids = self._eventHandler.roi_factory(self.ax)
            self._update_help(1)
        elif self._active == 'brush':
            self._ids = self._eventHandler.brush_factory(self.ax)
            self._update_help(2)
        else:
            self._ids = self._eventHandler.zoom_pan_factory(self.ax)
            self._update_help(0)

    def _on_reset(self):
        # clear bad pixels to restore a clean state
        # TODO: investigate why self.clear_overlay() is not working
        self.overlay = None
        self.set_overlay([], [])

        # clear all ROIs (red and blue)
        for sp in self._roi_all:
            sp.valueChanged.disconnect(self._update_roi_canvas)
        self._eventHandler.roi_changed.disconnect(self._update_roi)

        for sp in self._roi_all:
            sp.setValue(0.)

        self._eventHandler.ref_rect = None
        self._eventHandler.ref_idx = -1
        for rect in self._eventHandler.all_rect:
            rect.remove()
        if self._eventHandler.cross is not None:
            self._eventHandler.cross.remove()
            self._eventHandler.cross = None
        self._eventHandler.all_rect = []
        #self.ax.figure.canvas.draw()
        #self.canvas.draw()

        for sp in self._roi_all:
            sp.valueChanged.connect(self._update_roi_canvas)
        self._eventHandler.roi_changed.connect(self._update_roi)

        if self.image_handler:
            width = self.image.shape[1]
            height = self.image.shape[0]
            self.ax.set_xlim(0, width)
            self.ax.set_ylim(height, 0)
            self.canvas.draw()
    
    def _on_init_roi(self,roi):
        self._eventHandler.set_curr_roi(self.ax, (roi[0],roi[1]), roi[2], roi[3])
        self._update_roi(*roi)

    def _on_adjust_roi(self):
        '''
        Always use original image data (i.e. not log scaled one) for roi prediction.
        Also, currently, it ignores user selected (red-colored) roi
        todo: adjust based on user selected roi
        '''
        if self.image is None:
            return

        _x0, _y0, _w, _h = estimate_roi(self.image, threshold=1.0)
        cx = int(np.round(_x0 + _w//2))
        cy = int(np.round(_y0 + _h//2))

        side = 32 * (np.maximum(_w, _h) // 32)
        x0 = int(np.maximum(cx - side//2, 0))
        y0 = int(np.maximum(cy - side//2, 0))
        x1 = x0 + side
        y1 = y0 + side

        offset_x = np.maximum(x1 - self.image.shape[1] + 1, 0)
        x1 = x1 - offset_x

        offset_y = np.maximum(y1 - self.image.shape[0] + 1, 0)
        y1 = y1 - offset_y

        h = y1 - y0
        w = x1 - x0

        self._eventHandler.set_curr_roi(self.ax, (x0, y0), w, h)
        self._update_roi(x0, y0, w, h)

    def reset(self):
        for sp in self._roi_all:
            sp.setValue(0.)
        self.image = None
        self.image_data = None
        self.image_handler = None
        self.overlay = None
        self.overlay_handler = None
        self.ax.clear()
        self.ax.set_axis_off()
        self.canvas.draw()

    def draw_image(self, image, cmap='gray', init_roi=None, use_log=False):
        # TODO: merge this function and use_logscale()
        #print(cmap, init_roi, use_log)
        if use_log:
            image_data = np.nan_to_num(np.log(image + 1.))
        else:
            image_data = image

        if self.image_handler is None:
            self.image_handler = self.ax.imshow(image_data, cmap=cmap)
        else:
            self.image_handler.set_data(image_data)
            # todo: update data min, max (maybe not needed)
        self.image_handler.set_clim(vmin=np.min(image_data), vmax=np.max(image_data))

        self.image = image
        self.image_data = image_data

        if init_roi is not None:
            if not any(init_roi):
                self._on_adjust_roi()
            else:
                self._on_init_roi(init_roi)

        if len(self._ids) == 0:
            self._ids = self._eventHandler.zoom_pan_factory(self.ax)

        self.canvas.draw()

    def update_overlay(self, pixel, onoff):
        '''
        Update overlay image from brushed pixels
        '''
        if self.image is None: return

        self.show_overlay(True)
        
        if self.overlay is None or self.overlay.shape[:2] != self.image.shape:
            self.overlay = np.zeros(self.image.shape + (4,), dtype=np.float32)
        
        highlight = (1., 0., 1., .5)
        x, y = pixel
        #if self.overlay[y, x, 0] == 1.:
        #    self.overlay[y, x] = (0., 0., 0., 0.)
        if onoff:
            self.overlay[y, x] = highlight
        else:
            self.overlay[y, x] = (0., 0., 0., 0.)

        if self.overlay_handler is None:
            self.overlay_handler = self.ax.imshow(self.overlay)
        else:
            self.overlay_handler.set_data(self.overlay)
            self.overlay_handler.set_visible(True)
            self.show_overlay(True)
            # todo: set show badpixel flag
        self.canvas.draw()


    def set_overlay(self, rows, cols):
        if self.image is None: return
        if len(rows) != len(cols): return

        self.show_overlay(True)

        highlight = (1., 0, 1., .5)
        if self.overlay is None:
            self.overlay = np.zeros(self.image.shape + (4,), dtype=np.float32)

        self.overlay[rows, cols] = highlight
        if self.overlay_handler is None:
            self.overlay_handler = self.ax.imshow(self.overlay)
        else:
            self.overlay_handler.set_data(self.overlay)
        self.overlay_handler.set_visible(True)
        self.canvas.draw()

    def clear_overlay(self):
        if self.overlay is None: return
        self.overlay[:,:,0] = 0
        self.canvas.draw()

    def show_overlay(self, state):
        if self.overlay_handler is None: return
        self.overlay_handler.set_visible(state)
        self.canvas.draw()

    def use_logscale(self, state):
        # TODO: merge this function and draw_image()
        if self.image is None: return
        if state:
            self.image_data = np.log(np.clip(self.image, 1., None))
        else:
            self.image_data = self.image
        self.image_handler.set_data(self.image_data)
        self.image_handler.set_clim(vmin=np.min(self.image_data), vmax=np.max(self.image_data))
        self.canvas.draw()

    def get_red_roi(self):
        '''
        Return red colored ROI.
        If there are multiple, return the largest area one
        '''
        all_roi = self._eventHandler.get_red_roi()

        largest_roi = None
        largest_area = 0.

        for roi in all_roi:
            xy, width, height = roi
            # canonicalize the ROI
            x0, y0 = xy
            if width < 0:
                x0 += width
                width = -width
            if height < 0:
                y0 += height
                height = -height
            area = width * height
            if area > largest_area:
                largest_area = area
                largest_roi = (
                    int(np.floor(x0 + 0.5)),
                    int(np.floor(y0 + 0.5)),
                    int(np.round(width)),
                    int(np.round(height))
                )

        return largest_roi

    def get_blue_roi(self):
        '''
        Return blue colored ROI
        '''
        all_roi = []
        for roi in self._eventHandler.get_blue_roi():
            xy, width, height = roi
            # canonicalize the ROI
            x0, y0 = xy
            if width < 0:
                x0 += width
                width = -width
            if height < 0:
                y0 += height
                height = -height
            all_roi.append((
                int(np.floor(x0 + 0.5)),
                int(np.floor(y0 + 0.5)),
                int(np.round(width)),
                int(np.round(height))
            ))

        return all_roi

    def get_badpixels(self):
        if self.overlay is None: return None
        return np.where(self.overlay[:,:,0])

