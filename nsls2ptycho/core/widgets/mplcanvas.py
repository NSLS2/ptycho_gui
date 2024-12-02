import os
from PyQt5 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_area_auto_adjustable

import csv
import numpy as np
from PIL import Image

from ..ptycho.utils import split


def load_image_pil(path):
    """
    Read images using the PIL lib
    """
    file = Image.open(str(path))  # 'I;16B'
    return np.array(file.getdata()).reshape(file.size[::-1])


def load_image_ascii(path):
    """
    Read ASCII images using the csv lib
    """
    delimiter = '\t'
    data = []
    for row in csv.reader(open(path), delimiter=delimiter):
        data.append(row[:-1])
    img = np.array(data).astype(np.double)
    return img


def brush_to_color_tuple(brush):
    r, g, b, a = brush.color().getRgbF()
    return r, g, b


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        ax = Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        self.axes = ax
        self.fig = fig

        self.canvas = FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        #self.figure.subplots_adjust(left= 0.15, bottom=0.15)
        window_brush = self.window().palette().window()
        fig.set_facecolor(brush_to_color_tuple(window_brush))
        fig.set_facecolor(brush_to_color_tuple(window_brush))

        self.reset()

    def reset(self):
        self.image_handlers = None
        self.line_handlers = []
        self.axes.clear()
        self.axes.set_axis_off() # must be after clear()
        self.draw() # test 

    def compute_initial_figure(self):
        pass

    def axis_on(self):
        self.axes.set_axis_on()
        make_axes_area_auto_adjustable(self.axes)

    def update_image(self, image):
        if self.image_handlers is None:
            self.image_handlers = self.axes.imshow(image)
        else:
            self.image_handlers.set_data(image)
            # amplitude and phase have dramatically different ranges, so rescaling is necessary
            self.image_handlers.autoscale()
        self.draw()

    def update_plot(self, xValues, yValues):

        num_plots = yValues.shape[1]
        if len(self.line_handlers) == 0:
            for i in range(num_plots):
                h = self.axes.semilogy(xValues, yValues[:,i])
                self.line_handlers.append(h[0])
        else:
            for hidx, h in enumerate(self.line_handlers):
                h.set_data(xValues, yValues[:,hidx])
            self.axes.tick_params(axis='both', labelsize=8)
            self.axes.set_yscale('log')
            self.axes.relim(visible_only=True)
            #self.axes.autoscale(tight=True)
            #self.axes.autoscale_view(tight=True)
            self.axes.autoscale_view()

        self.draw()

    def update_scatter(self, pts, mpi_size, colormap=cm.jet):
        '''
        Plot N scanning points in mpi_size different colors

        Parameters:
            - pts: np.array([[x0, x1, ..., xN], [y0, y1, ..., yN]])
            - mpi_size: number of MPI processes
            - colormap 
        '''
        # only show up to 15 items in the legend to fit in the window
        label_set = set([i  for i in range(9)] + [i  for i in range(mpi_size, mpi_size-6, -1)])
        #labels = []
        too_long = False
        a = split(pts.shape[1], mpi_size)
        if len(self.line_handlers) == 0:
            colors = colormap(np.linspace(0, 1, len(a)))
            for i in range(mpi_size):
                if mpi_size <=15 or i in label_set:
                    label = 'Process %i'%i
                    #labels.append(label)
                elif i==mpi_size-6 and i not in label_set:
                    label = r'    $\vdots$'
                    #labels.append(label)
                    too_long = True
                else:
                    label = '_nolegend_' # matplotlib undocumented secret...
                h = self.axes.scatter(pts[0, a[i][0]:a[i][1]], pts[1, a[i][0]:a[i][1]], c=[colors[i]], label=label)
                self.line_handlers.append(h)
        else: # assuming mpi_size is unchanged
            for i, h in enumerate(self.line_handlers):
                h.set_offsets(np.array([pts[0, a[i][0]:a[i][1]], pts[1, a[i][0]:a[i][1]]]).transpose())
            # TODO: handle plot limits?
            ##self.axes.tick_params(axis='both', labelsize=8)
            #self.axes.relim(visible_only=True)
            ##self.axes.autoscale(tight=True)
            ##self.axes.autoscale_view(tight=True)
            #self.axes.autoscale_view()

        # we have a rectangular window, make the plot align to its center left
        self.axes.set_aspect(aspect='equal', anchor='W')
        legend = self.axes.legend(bbox_to_anchor=(0.98, 1.0), fancybox=True)

        # for the label \vdots, remove its marker
        if too_long:
            legend.legendHandles[9].set_sizes([0])
            #self.axes.legend(legend.legendHandles, labels, bbox_to_anchor=(0.98, 1.0), fancybox=True)

        self.draw()
