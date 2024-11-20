import sys
import os
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QAction

from nsls2ptycho.ui import ui_ptycho
from nsls2ptycho.core.utils import clean_shared_memory, get_mpi_num_processes, parse_range
from nsls2ptycho.core.ptycho_param import Param
from nsls2ptycho.core.ptycho_recon import PtychoReconWorker, PtychoReconFakeWorker, HardWorker
from nsls2ptycho.core.ptycho_qt_utils import PtychoStream
from nsls2ptycho.core.widgets.list_widget import ListWidget
from nsls2ptycho.core.widgets.mplcanvas import load_image_pil
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho._version import __version__

# databroker related
from nsls2ptycho.core.databroker_api import db, load_metadata, get_single_image, get_detector_names, beamline_name

from nsls2ptycho.reconStep_gui import ReconStepWindow
from nsls2ptycho.roi_gui import RoiWindow
from nsls2ptycho.scan_pt import ScanWindow


import h5py
import numpy as np
from numpy import pi
import traceback

# for frontend-backend communication
from posix_ipc import SharedMemory, ExistentialError
import mmap


# set True for testing GUI changes
_TEST = False

# for shared memory
mm_list = []
shm_list = []


class MainWindow(QtWidgets.QMainWindow, ui_ptycho.Ui_MainWindow):
    _mainwindow_signal = QtCore.pyqtSignal()

    def __init__(self, parent=None, param:Param=None):
        super().__init__(parent)
        self.setupUi(self)
        QtWidgets.QApplication.setStyle('Plastique')

        # connect
        self.btn_load_probe.clicked.connect(self.loadProbe)
        self.btn_load_object.clicked.connect(self.loadObject)
        self.ck_init_prb_flag.clicked.connect(self.resetProbeFlg)
        self.ck_init_obj_flag.clicked.connect(self.resetObjectFlg)

        self.btn_choose_cwd.clicked.connect(self.setWorkingDirectory)
        self.cb_dataloader.currentTextChanged.connect(self.setLoadButton)
        self.btn_load_scan.clicked.connect(self.loadExpParam)
        self.btn_view_frame.clicked.connect(self.viewDataFrame)
        self.ck_extra_scans_flag.clicked.connect(self.updateExtraScansFlg)
        self.btn_set_extra_scans.clicked.connect(self.setExtraScans)

        #self.le_scan_num.editingFinished.connect(self.forceLoad) # too sensitive, why?
        self.le_scan_num.textChanged.connect(self.forceLoad)
        self.cb_dataloader.currentTextChanged.connect(self.forceLoad)
        self.cb_detectorkind.currentTextChanged.connect(self.forceLoad)

        self.ck_mode_flag.clicked.connect(self.modeMultiSliceGuard)
        self.ck_multislice_flag.clicked.connect(self.modeMultiSliceGuard)
        self.ck_mask_obj_flag.clicked.connect(self.updateObjMaskFlg)
        self.ck_gpu_flag.clicked.connect(self.updateGpuFlg)
        self.ck_bragg_flag.clicked.connect(self.updateBraggFlg)
        self.ck_pc_flag.clicked.connect(self.updatePcFlg)
        self.ck_position_correction_flag.clicked.connect(self.updateCorrFlg)
        self.ck_refine_data_flag.clicked.connect(self.updateRefineDataFlg)
        self.ck_postprocessing_flag.clicked.connect(self.showNoPostProcessingWarning)
        self.ck_batch_crop_flag.clicked.connect(self.updateBatchCropDataFlg)
        self.cb_dataloader.currentTextChanged.connect(self.updateBatchCropDataFlg)

        self.btn_recon_start.clicked.connect(self.start)
        self.btn_recon_stop.clicked.connect(self.stop)
        self.btn_recon_batch_start.clicked.connect(self.batchStart)
        self.btn_recon_batch_stop.clicked.connect(self.batchStop)
        self.ck_init_prb_batch_flag.stateChanged.connect(self.switchProbeBatch)
        self.ck_init_obj_batch_flag.stateChanged.connect(self.switchObjectBatch)

        self.menu_import_config.triggered.connect(self.importConfig)
        self.menu_export_config.triggered.connect(self.exportConfig)
        self.menu_clear_config_history.triggered.connect(self.removeConfigHistory)
        self.menu_save_config_history.triggered.connect(self.saveConfigHistory)
        self.actionClear_shared_memory.triggered.connect(self.clearSharedMemory)

        self.btn_MPI_file.clicked.connect(self.setMPIfile)
        self.le_gpus.textChanged.connect(self.resetMPIFlg)

        # setup
        self.sp_pha_max.setMaximum(pi)
        self.sp_pha_max.setMinimum(-pi)
        self.sp_pha_min.setMaximum(pi)
        self.sp_pha_min.setMinimum(-pi)

        # init.
        if param is None:
            self.param = Param() # default
        else:
            self.param = param
        self._prb = None
        self._obj = None
        self._ptycho_gpu_thread = None
        self._worker_thread = None
        self._db = None             # hold the Broker instance that contains the info of the given scan id
        self._mds_table = None      # hold a Pandas.dataframe instance
        self._loaded = False        # whether the user has loaded metadata or not (from either databroker or h5)
        self._scan_numbers = None   # a list of scan numbers for batch mode
        self._scan_points = None    # an array of shape (2, N) holding the scan coordinates
        self._extra_scans_dialog = None
        self._batch_prb_filename = None  # probe's filename template for batch mode
        self._batch_obj_filename = None  # object's filename template for batch mode
        self._config_path = os.path.expanduser("~") + "/.ptycho_gui/.ptycho_gui_config"
        if not os.path.isdir(os.path.dirname(self._config_path)):
            os.makedirs(os.path.dirname(self._config_path))

        self.reconStepWindow = None
        self.roiWindow = None
        self.scanWindow = None

        # temporary solutions
        self.ck_ms_pie_flag.setEnabled(False)
        self.ck_weak_obj_flag.setEnabled(False)
        #self.cb_alg_flag. addItem("PIE")
        #self.cb_alg2_flag.addItem("PIE")
        # TODO: find a way to register the live windows so that they can be opened anytime
        self.menuWindows.setEnabled(False)
        #self.actionROI.setEnabled(False)
        #self.actionMonitor.setEnabled(False)
        #self.actionScan_points.setEnabled(False)

        #if self.menu_save_config_history.isChecked(): # TODO: think of a better way...
        self.retrieveConfigHistory()
        self.update_gui_from_param()
        self.updateExtraScansFlg()
        self.updateModeFlg()
        self.updateMultiSliceFlg()
        self.updateObjMaskFlg()
        self.updateBraggFlg()
        self.updatePcFlg()
        self.updateCorrFlg()
        self.updateRefineDataFlg()
        self.updateBatchCropDataFlg()
        self.checkGpuAvail()
        self.updateGpuFlg()
        self.resetExperimentalParameters() # probably not necessary
        self.setLoadButton()

        # generate a unique string for shared memory
        if sys.platform.startswith('darwin'): # OS X has a much shorter name limit
            self.param.shm_name = os.getlogin()+'_'+str(os.getpid())+'_'+str(random.randrange(256))
        else:
            self.param.shm_name = 'ptycho_'+os.getlogin()+'_'+str(os.getpid())+'_'+str(random.randrange(256))

        # TODO: delete param.shm_name read in from previous config so that we can reset the buttons earlier
        self.resetButtons()

        # display GUI version
        self.setWindowTitle("NSLS-II Ptychography v" + __version__)


    @property
    def db(self):
        # access the Broker instance; the name is probably not intuitive enough...?
        return self._db


    @db.setter
    def db(self, scan_id:int):
        # TODO: this should be configured based on selected beamline profile!
        self._db = db


    def resetButtons(self):
        self.btn_recon_start.setEnabled(True)
        self.btn_recon_stop.setEnabled(False)
        self.btn_recon_batch_start.setEnabled(True)
        self.btn_recon_batch_stop.setEnabled(False)
        self.recon_bar.setValue(0)
        # close the mmap arrays
        # removing these arrays, can be changed later if needed
        if self._prb is not None:
            del self._prb
            self._prb = None
        if self._obj is not None:
            del self._obj
            self._obj = None
        #if self._scan_points is not None:
        #    del self._scan_points
        #    self._scan_points = None
        self.close_mmap()

    
    # TODO: consider merging this function with importConfig()? 
    def retrieveConfigHistory(self):
        if os.path.isfile(self._config_path):
            try:
                param = parse_config(self._config_path, Param())
                self.menu_save_config_history.setChecked(param.save_config_history)
                if param.save_config_history:
                    self.param = param
            except Exception as ex:
                self.exception_handler(ex)


    def saveConfigHistory(self):
        self.param.save_config_history = self.menu_save_config_history.isChecked()


    def removeConfigHistory(self):
        if os.path.isfile(self._config_path):
            self.param = Param() # default
            os.remove(self._config_path)
            self.update_gui_from_param()
        

    def update_param_from_gui(self):
        p = self.param

        # data group
        p.scan_num = str(self.le_scan_num.text())
        p.detectorkind = str(self.cb_detectorkind.currentText())
        p.frame_num = int(self.sp_fram_num.value())
        # p.working_directory set by setWorkingDirectory()

        # Exp param group
        p.xray_energy_kev = float(self.sp_xray_energy.value())
        if self.sp_xray_energy.value() != 0.:
            p.lambda_nm = 1.2398/self.sp_xray_energy.value()
        p.z_m = float(self.sp_detector_distance.value())
        p.nx = int(self.sp_x_arr_size.value()) # bookkeeping
        p.dr_x = float(self.sp_x_step_size.value())
        p.x_range = float(self.sp_x_scan_range.value())
        p.ny = int(self.sp_y_arr_size.value()) # bookkeeping
        p.dr_y = float(self.sp_y_step_size.value())
        p.y_range = float(self.sp_y_scan_range.value())
        #p.scan_type = str(self.cb_scan_type.currentText()) # do we need this one?
        p.nz = int(self.sp_num_points.value()) # bookkeeping

        # recon param group 
        p.n_iterations = int(self.sp_n_iterations.value())
        p.alg_flag = str(self.cb_alg_flag.currentText())
        p.alg2_flag = str(self.cb_alg2_flag.currentText())
        p.alg_percentage = float(self.sp_alg_percentage.value())
        p.sign = str(self.le_sign.text())
        p.precision = self.cb_precision_flag.currentText()

        p.init_prb_flag = self.ck_init_prb_flag.isChecked()
        p.init_obj_flag = self.ck_init_obj_flag.isChecked()
        # prb and obj path already set 

        p.mode_flag = self.ck_mode_flag.isChecked()
        p.prb_mode_num = self.sp_prb_mode_num.value()
        p.obj_mode_num = self.sp_obj_mode_num.value()
        if p.mode_flag and "_mode" not in p.sign:
            p.sign = p.sign + "_mode"

        p.multislice_flag = self.ck_multislice_flag.isChecked()
        p.slice_num = int(self.sp_slice_num.value())
        p.slice_spacing_m = float(self.sp_slice_spacing_m.value() * 1e-6)
        if p.multislice_flag and "_ms" not in p.sign:
            p.sign = p.sign + "_ms"

        p.amp_min = float(self.sp_amp_min.value())
        p.amp_max = float(self.sp_amp_max.value())
        p.pha_min = float(self.sp_pha_min.value())
        p.pha_max = float(self.sp_pha_max.value())

        p.gpu_flag = self.ck_gpu_flag.isChecked()
        p.gpus = parse_range(self.le_gpus.text(), batch_processing=False)
        p.gpu_batch_size = int(self.cb_gpu_batch_size.currentText())

        # adv param group
        p.ccd_pixel_um = float(self.sp_ccd_pixel_um.value())
        p.distance = float(self.sp_distance.value())
        p.angle_correction_flag = self.ck_angle_correction_flag.isChecked()
        p.x_direction = float(self.sp_x_direction.value())
        p.y_direction = float(self.sp_y_direction.value())
        p.angle = self.sp_angle.value()

        p.start_update_probe = self.sp_start_update_probe.value()
        p.start_update_object = self.sp_start_update_object.value()
        p.ml_mode = self.cb_ml_mode.currentText()
        p.ml_weight = self.sp_ml_weight.value()
        p.dm_version = self.sp_dm_version.value()
        p.cal_scan_pattern_flag = self.ck_cal_scal_pattern_flag.isChecked()
        p.nth = self.sp_nth.value()
        p.start_ave = self.sp_start_ave.value()
        p.processes = self.sp_processes.value()

        p.bragg_flag = self.ck_bragg_flag.isChecked()
        p.bragg_theta = self.sp_bragg_theta.value()
        p.bragg_gamma = self.sp_bragg_gamma.value()
        p.bragg_delta = self.sp_bragg_delta.value() 

        p.pc_flag = self.ck_pc_flag.isChecked()
        p.pc_sigma = self.sp_pc_sigma.value()
        p.pc_alg = self.cb_pc_alg.currentText()
        p.pc_kernel_n = self.sp_pc_kernel_n.value()

        p.position_correction_flag = self.ck_position_correction_flag.isChecked()
        p.position_correction_start = self.sp_position_correction_start.value()
        p.position_correction_step = self.sp_position_correction_step.value()  

        p.sigma2 = float(self.sp_sigma2.value())
        p.beta = float(self.sp_beta.value())
        p.display_interval = int(self.sp_display_interval.value())
        p.preview_flag = self.ck_preview_flag.isChecked()
        p.cal_error_flag = self.ck_cal_error_flag.isChecked()

        p.prb_center_flag = self.ck_prb_center_flag.isChecked()
        p.mask_obj_flag = self.ck_mask_obj_flag.isChecked()
        p.norm_prb_amp_flag = self.ck_norm_prb_amp_flag.isChecked()
        p.weak_obj_flag = self.ck_weak_obj_flag.isChecked()
        p.ms_pie_flag = self.ck_ms_pie_flag.isChecked()

        p.refine_data_flag     = self.ck_refine_data_flag.isChecked()
        p.refine_data_start_it = int(self.sp_refine_data_start_it.value())
        p.refine_data_interval = int(self.sp_refine_data_interval.value())
        p.refine_data_step     = float(self.sp_refine_data_step.value())

        p.profiler_flag        = self.ck_profiler_flag.isChecked()
        p.postprocessing_flag  = self.ck_postprocessing_flag.isChecked()
        p.use_NCCL             = self.rb_nccl.isChecked()
        p.use_CUDA_MPI         = self.rb_cuda_mpi.isChecked()

        # TODO: organize them
        #self.ck_init_obj_dpc_flag.setChecked(p.init_obj_dpc_flag) 
        #self.ck_mask_prb_flag.setChecked(p.mask_prb_flag)
        #self.ck_mesh_flag.setChecked(p.mesh_flag)
        #self.ck_sf_flag.setChecked(p.sf_flag)

        # batch param group, necessary?

        # from the associate scan number window
        if self._extra_scans_dialog is not None:
            scans = self._extra_scans_dialog.listWidget
            num_items = scans.count()
            p.asso_scan_numbers = [scans.item(i).text() for i in range(num_items)]
        else:
            # do not erase this, as keeping it has no harm
            pass


    def update_gui_from_param(self):
        p = self.param

        # Data group
        self.le_scan_num.setText(p.scan_num)
        self.le_working_directory.setText(str(p.working_directory or ''))
        self.cb_detectorkind.setCurrentIndex(p.get_detector_kind_index())
        self.sp_fram_num.setValue(int(p.frame_num))

        # Exp param group
        self.sp_xray_energy.setValue(1.2398/float(p.lambda_nm) if 'lambda_nm' in p.__dict__ else 0.)
        self.sp_detector_distance.setValue(float(p.z_m) if 'z_m' in p.__dict__ else 0)
        self.sp_x_arr_size.setValue(float(p.nx))
        self.sp_x_step_size.setValue(float(p.dr_x))
        self.sp_x_scan_range.setValue(float(p.x_range))
        self.sp_y_arr_size.setValue(float(p.ny))
        self.sp_y_step_size.setValue(float(p.dr_y))
        self.sp_y_scan_range.setValue(float(p.y_range))
        self.cb_scan_type.setCurrentIndex(p.get_scan_type_index())
        self.sp_num_points.setValue(int(p.nz))

        # recon param group
        self.sp_n_iterations.setValue(int(p.n_iterations))
        self.cb_alg_flag.setCurrentIndex(p.get_alg_flg_index())
        self.cb_alg2_flag.setCurrentIndex(p.get_alg2_flg_index())
        self.sp_alg_percentage.setValue(float(p.alg_percentage))
        self.le_sign.setText(p.sign)
        self.cb_precision_flag.setCurrentText(p.precision)

        self.ck_init_prb_flag.setChecked(p.init_prb_flag)
        self.le_prb_path.setText(str(p.prb_filename or ''))

        self.ck_init_obj_flag.setChecked(p.init_obj_flag)
        self.le_obj_path.setText(str(p.obj_filename or ''))

        self.ck_mode_flag.setChecked(p.mode_flag)
        self.sp_prb_mode_num.setValue(int(p.prb_mode_num))
        self.sp_obj_mode_num.setValue(int(p.obj_mode_num))

        self.ck_multislice_flag.setChecked(p.multislice_flag)
        self.sp_slice_num.setValue(int(p.slice_num))
        self.sp_slice_spacing_m.setValue(p.get_slice_spacing_m())

        self.sp_amp_max.setValue(float(p.amp_max))
        self.sp_amp_min.setValue(float(p.amp_min))
        self.sp_pha_max.setValue(float(p.pha_max))
        self.sp_pha_min.setValue(float(p.pha_min))

        self.ck_gpu_flag.setChecked(p.gpu_flag)
        gpu_str = ''
        for i, dev_id in enumerate(p.gpus):
            gpu_str += str(dev_id)
            if i != len(p.gpus) - 1:
                gpu_str += ', '
        self.le_gpus.setText(gpu_str)
        self.cb_gpu_batch_size.setCurrentIndex(p.get_gpu_batch_index())
        
        # set MPI file path from param    
        if p.mpi_file_path != '':
            mpi_filename = os.path.basename(p.mpi_file_path)
            self.le_MPI_file_path.setText(mpi_filename)
            # TODO: does this make sense?
            self.le_gpus.setText('')

        # adv param group
        self.sp_ccd_pixel_um.setValue(p.ccd_pixel_um)
        self.sp_distance.setValue(float(p.distance))
        self.ck_angle_correction_flag.setChecked(p.angle_correction_flag)
        self.sp_x_direction.setValue(p.x_direction)
        self.sp_y_direction.setValue(p.y_direction)
        self.sp_angle.setValue(p.angle)

        self.sp_start_update_probe.setValue(p.start_update_probe)
        self.sp_start_update_object.setValue(p.start_update_object)
        self.cb_ml_mode.setCurrentText(p.ml_mode)
        self.sp_ml_weight.setValue(p.ml_weight)
        self.sp_dm_version.setValue(p.dm_version)
        self.ck_cal_scal_pattern_flag.setChecked(p.cal_scan_pattern_flag)
        self.sp_nth.setValue(p.nth)
        self.sp_start_ave.setValue(p.start_ave)
        self.sp_processes.setValue(p.processes)

        self.ck_bragg_flag.setChecked(p.bragg_flag)
        self.sp_bragg_theta.setValue(p.bragg_theta)
        self.sp_bragg_gamma.setValue(p.bragg_gamma)
        self.sp_bragg_delta.setValue(p.bragg_delta)

        self.ck_pc_flag.setChecked(p.pc_flag)
        self.sp_pc_sigma.setValue(p.pc_sigma)
        self.cb_pc_alg.setCurrentText(p.pc_alg)
        self.sp_pc_kernel_n.setValue(p.pc_kernel_n)

        self.ck_position_correction_flag.setChecked(p.position_correction_flag)
        self.sp_position_correction_start.setValue(p.position_correction_start)
        self.sp_position_correction_step.setValue(p.position_correction_step)

        self.sp_sigma2.setValue(p.sigma2)
        self.sp_beta.setValue(p.beta)
        self.sp_display_interval.setValue(p.display_interval)
        self.ck_preview_flag.setChecked(p.preview_flag)
        self.ck_cal_error_flag.setChecked(p.cal_error_flag)

        self.ck_init_obj_dpc_flag.setChecked(p.init_obj_dpc_flag) 
        self.ck_prb_center_flag.setChecked(p.prb_center_flag)
        self.ck_mask_prb_flag.setChecked(p.mask_prb_flag)
        self.ck_mask_obj_flag.setChecked(p.mask_obj_flag)
        self.ck_norm_prb_amp_flag.setChecked(p.norm_prb_amp_flag)
        self.ck_weak_obj_flag.setChecked(p.weak_obj_flag)
        self.ck_mesh_flag.setChecked(p.mesh_flag)
        self.ck_ms_pie_flag.setChecked(p.ms_pie_flag)
        self.ck_sf_flag.setChecked(p.sf_flag)

        self.ck_refine_data_flag.setChecked(p.refine_data_flag)
        self.sp_refine_data_start_it.setValue(p.refine_data_start_it)
        self.sp_refine_data_interval.setValue(p.refine_data_interval)
        self.sp_refine_data_step.setValue(p.refine_data_step)

        self.ck_profiler_flag.setChecked(p.profiler_flag)
        self.ck_postprocessing_flag.setChecked(p.postprocessing_flag)
        self.rb_nccl.setChecked(p.use_NCCL)
        self.rb_cuda_mpi.setChecked(p.use_CUDA_MPI)

        # batch param group, necessary?


    def start(self, batch_mode=False):
        if self._ptycho_gpu_thread is not None and self._ptycho_gpu_thread.isFinished():
            self._ptycho_gpu_thread = None

        if self._ptycho_gpu_thread is None:
            if not self._loaded:
                print("[WARNING] Remember to click \"Load\" before proceeding!", file=sys.stderr) 
                return

            self.update_param_from_gui() # this has to be done first, so all operations depending on param are correct
            self.recon_bar.setValue(0)
            self.recon_bar.setMaximum(self.param.n_iterations)

            # at least one GPU needs to be selected
            if self.param.gpu_flag and len(self.param.gpus) == 0 and self.param.mpi_file_path == '':
                print("[WARNING] select at least one GPU!", file=sys.stderr)
                return

            # batch mode requires some additional changes to param
            if batch_mode:
                if self._batch_prb_filename is not None:
                    p = self.param
                    p.init_prb_flag = False
                    scan_num = str(self.param.scan_num)
                    sign = self._batch_prb_filename[1].split('probe')[0]
                    sign = sign.strip('_')
                    dirname = p.working_directory + "/recon_result/S" + scan_num + "/" + sign + "/recon_data/"
                    filename = scan_num.join(self._batch_prb_filename)
                    p.set_prb_path(dirname, filename)
                    print("[BATCH] will load " + dirname + filename + " as probe")

                if self._batch_obj_filename is not None:
                    p = self.param
                    p.init_obj_flag = False
                    scan_num = str(self.param.scan_num)
                    sign = self._batch_obj_filename[1].split('object')[0]
                    sign = sign.strip('_')
                    dirname = p.working_directory + "/recon_result/S" + scan_num + "/" + sign + "/recon_data/"
                    filename = scan_num.join(self._batch_obj_filename)
                    p.set_obj_path(dirname, filename)
                    print("[BATCH] will load " + dirname + filename + " as object")

            # this is needed because MPI processes need to know the working directory...
            self._exportConfigHelper(self._config_path)

            # init reconStepWindow
            if self.ck_preview_flag.isChecked():
                if self.param.mode_flag:
                    info = (self.param.obj_mode_num, self.param.prb_mode_num, 1)
                elif self.param.multislice_flag:
                    info = (self.param.slice_num, 1, 1)
                else: 
                    info = (1, 1, 2)

                if self.reconStepWindow is None:
                    self.reconStepWindow = ReconStepWindow(*info)
                self.reconStepWindow.reset_window(*info, iterations=self.param.n_iterations,
                                                  slider_interval=self.param.display_interval)
                self.reconStepWindow.show()
            else:
                if self.reconStepWindow is not None:
                    # TODO: maybe a thorough cleanup???
                    self.reconStepWindow.close()

            if not _TEST:
                thread = self._ptycho_gpu_thread = PtychoReconWorker(self.param, parent=self)
            else:
                thread = self._ptycho_gpu_thread = PtychoReconFakeWorker(self.param, parent=self)

            thread.update_signal.connect(self.update_recon_step)
            thread.finished.connect(self.resetButtons)
            if batch_mode:
                thread.finished.connect(self._batch_manager)
            #thread.finished.connect(self.reconStepWindow.debug)
            thread.start()

            self.btn_recon_stop.setEnabled(True)
            self.btn_recon_start.setEnabled(False)

            # init scan window
            # TODO: optimize and refactor this part
            if self.ck_scan_pt_flag.isChecked():
                if self.scanWindow is None:
                    self.scanWindow = ScanWindow()
                    self.scanWindow.reset_window()
                    self.scanWindow.show()
            else:
                if self.scanWindow is not None:
                    self.scanWindow.close()
                    self.scanWindow = None
                return

            if self._scan_points is None:
                raise RuntimeError("Scan points were not read. This shouldn't happen. Abort.")
            else:
                self._scan_points[0] *= -1.*self.param.x_direction
                self._scan_points[1] *= self.param.y_direction
                # borrowed from nsls2ptycho/core/ptycho_recon.py
                if self.param.mpi_file_path == '':
                    if self.param.gpu_flag:
                        num_processes = str(len(self.param.gpus))
                    else:
                        num_processes = str(self.param.processes) if self.param.processes > 1 else str(1)
                else:
                    # regardless if GPU is used or not --- trust users to know this
                    num_processes = str(get_mpi_num_processes(self.param.mpi_file_path))
                self.scanWindow.update_image(self._scan_points, int(num_processes))


    def stop(self, batch_mode=False):
        if self._ptycho_gpu_thread is not None:
            if batch_mode:
                self._ptycho_gpu_thread.finished.disconnect(self._batch_manager)
            if self._ptycho_gpu_thread.isRunning():
                self._ptycho_gpu_thread.kill() # first kill the mpi processes
                self._ptycho_gpu_thread.quit() # then quit QThread gracefully
            self._ptycho_gpu_thread = None
            self.resetButtons()
            if self.reconStepWindow is not None:
                self.reconStepWindow.reset_window()
            if self.scanWindow is not None:
                self.scanWindow.reset_window()


    def init_mmap(self):
        p = self.param
        datasize = 8 if p.precision == 'single' else 16
        datatype = np.complex64 if p.precision == 'single' else np.complex128

        global mm_list, shm_list
        for i, name in enumerate(["/"+p.shm_name+"_obj_size", "/"+p.shm_name+"_prb", "/"+p.shm_name+"_obj"]):
            shm_list.append(SharedMemory(name))
            mm_list.append(mmap.mmap(shm_list[i].fd, shm_list[i].size))

        nx_obj = int.from_bytes(mm_list[0].read(8), byteorder='big')
        ny_obj = int.from_bytes(mm_list[0].read(8), byteorder='big') # the file position has been moved by 8 bytes when we get nx_obj

        if p.mode_flag:
            self._prb = np.ndarray(shape=(p.n_iterations, p.prb_mode_num, p.nx, p.ny), dtype=datatype, buffer=mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.obj_mode_num, nx_obj, ny_obj), dtype=datatype, buffer=mm_list[2], order='C')
        elif p.multislice_flag:
            self._prb = np.ndarray(shape=(p.n_iterations, 1, p.nx, p.ny), dtype=datatype, buffer=mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.slice_num, nx_obj, ny_obj), dtype=datatype, buffer=mm_list[2], order='C')
        else:
            self._prb = np.ndarray(shape=(p.n_iterations, 1, p.nx, p.ny), dtype=datatype, buffer=mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, 1, nx_obj, ny_obj), dtype=datatype, buffer=mm_list[2], order='C')


    def close_mmap(self):
        # We close shared memory as long as the backend is terminated either normally or 
        # abnormally. The subtlety here is that the monitor should still be able to access
        # the intermediate results after mmaps' are closed. A potential segfault is avoided 
        # by accessing the transformed results, which are buffered, not the original ones.
        try:
            global mm_list, shm_list
            for mm, shm in zip(mm_list, shm_list):
                mm.close()
                shm.close_fd()
                shm.unlink()
            mm_list = []
            shm_list = []
        except NameError:
            # either not using GUI, monitor is turned off, global variables are deleted or not yet created!
            # need to examine the last case
            try:
                SharedMemory("/"+self.param.shm_name+"_obj_size").unlink()
                SharedMemory("/"+self.param.shm_name+"_prb").unlink()
                SharedMemory("/"+self.param.shm_name+"_obj").unlink()
            except ExistentialError:
                pass # nothing to clean up, we're done


    def update_recon_step(self, it, data=None):
        self.recon_bar.setValue(it)

        if self.reconStepWindow is not None:
            self.reconStepWindow.update_iter(it)

            if not _TEST and self.ck_preview_flag.isChecked():
                try:
                    if it == -1 and data == 'init_mmap':
                        try:
                            # the two npy are created by ptycho by this time
                            self.init_mmap()
                        except ExistentialError:
                            # user may kill the process prematurely
                            self.stop()
                    elif it == self.param.n_iterations+1:
                        # reserve it=n_iterations+1 as the working space
                        self.reconStepWindow.current_max_iters = self.param.n_iterations

                        p = self.param
                        if not p.postprocessing_flag:
                            return
                        work_dir = p.working_directory
                        scan_num = str(p.scan_num)
                        data_dir = work_dir+'/recon_result/S'+scan_num+'/'+p.sign+'/recon_data/'
                        data = {}
                        images = []
                        print("[SUCCESS] generated results are loaded in the preview window. ", end='', file=sys.stderr)
                        print("Slide to frame "+str(p.n_iterations+1)+" and select from drop-down menus.", file=sys.stderr)

                        if self.param.mode_flag:
                            # load data that has been averaged + orthonormalized + phase-ramp removed
                            for i in range(self.param.obj_mode_num):
                                data['obj_'+str(i)] = np.load(data_dir+'recon_'+scan_num+'_'+p.sign+'_' \
                                                               +'object_mode_orth_ave_rp_mode_'+str(i)+'.npy')
                                self.reconStepWindow.cb_image_object.addItem("Object "+str(i)+" (orth_ave_rp)")
                                # hard-wire the padding values here...
                                images.append( np.rot90(np.angle(data['obj_'+str(i)][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )
                                images.append( np.rot90(np.abs(data['obj_'+str(i)][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )

                            for i in range(self.param.prb_mode_num):
                                data['prb_'+str(i)] = np.load(data_dir+'recon_'+scan_num+'_'+p.sign+'_' \
                                                               +'probe_mode_orth_ave_rp_mode_'+str(i)+'.npy')
                                self.reconStepWindow.cb_image_probe.addItem("Probe "+str(i)+" (orth_ave_rp)")
                                images.append( np.rot90(np.abs(data['prb_'+str(i)])) )
                                images.append( np.rot90(np.angle(data['prb_'+str(i)])) )
                        elif self.param.multislice_flag:
                            # load data that has been averaged + phase-ramp removed
                            for i in range(self.param.slice_num):
                                data['obj_'+str(i)] = np.load(data_dir+'recon_'+scan_num+'_'+p.sign+'_' \
                                                               +'object_ave_rp_ms_'+str(i)+'.npy')
                                self.reconStepWindow.cb_image_object.addItem("Object "+str(i)+" (ave_rp)")
                                # hard-wire the padding values here...
                                images.append( np.rot90(np.angle(data['obj_'+str(i)][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )
                                images.append( np.rot90(np.abs(data['obj_'+str(i)][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )

                            for i in range(self.param.slice_num):
                                data['prb_'+str(i)] = np.load(data_dir+'recon_'+scan_num+'_'+p.sign+'_' \
                                                               +'probe_ave_rp_ms_'+str(i)+'.npy')
                                self.reconStepWindow.cb_image_probe.addItem("Probe "+str(i)+" (ave_rp)")
                                images.append( np.rot90(np.abs(data['prb_'+str(i)])) )
                                images.append( np.rot90(np.angle(data['prb_'+str(i)])) )
                        else:
                            # load data (ave & ave_rp)
                            for sol in ['ave', 'ave_rp']:
                                for tar, target in zip(['obj', 'prb'], ['object', 'probe']):
                                    data[tar+'_'+sol] = np.load(data_dir+'recon_'+scan_num+'_'+p.sign+'_'+target+'_'+sol+'.npy')

                            # calculate images
                            for sol in ['ave', 'ave_rp']:
                                self.reconStepWindow.cb_image_object.addItem("Object  ("+sol+")")
                                # hard-wire the padding values here...
                                images.append( np.rot90(np.angle(data['obj_'+sol][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )
                                images.append( np.rot90(np.abs(data['obj_'+sol][(p.nx+30)//2:-(p.nx+30)//2, (p.ny+30)//2:-(p.ny+30)//2])) )

                            for sol in ['ave', 'ave_rp']:
                                self.reconStepWindow.cb_image_probe.addItem("Probe ("+sol+")")
                                images.append( np.rot90(np.abs(data['prb_'+sol])) )
                                images.append( np.rot90(np.angle(data['prb_'+sol])) )

                        self.reconStepWindow.update_images(it, images)
                    elif (it-1) % self.param.display_interval == 0:
                        if self.param.mode_flag:
                            images = []
                            for i in range(self.param.obj_mode_num):
                                images.append(np.rot90(np.angle(self._obj[it-1, i])))
                                images.append(np.rot90(np.abs(self._obj[it-1, i])))
                            for i in range(self.param.prb_mode_num):
                                images.append(np.rot90(np.abs(self._prb[it-1, i])))
                                images.append(np.rot90(np.angle(self._prb[it-1, i])))
                        elif self.param.multislice_flag:
                            images = []
                            for i in range(self.param.slice_num):
                                images.append(np.rot90(np.angle(self._obj[it-1, i])))
                                images.append(np.rot90(np.abs(self._obj[it-1, i])))
                            #TODO: decide which probe we'd like to present
                            images.append(np.rot90(np.abs(self._prb[it-1, 0])))
                            images.append(np.rot90(np.angle(self._prb[it-1, 0])))
                        else:
                            images = [np.rot90(np.angle(self._obj[it-1, 0])),
                                      np.rot90(np.abs(self._obj[it-1, 0]  )),
                                      np.rot90(np.abs(self._prb[it-1, 0]  )),
                                      np.rot90(np.angle(self._prb[it-1, 0]))]
                        self.reconStepWindow.update_images(it, images)
                        self.reconStepWindow.update_metric(it, data)

                except TypeError as ex: # when MPI processes are terminated, _prb and _obj are deleted and so not subscriptable 
                    pass
            else:
                # -------------------- Sungsoo version -------------------------------------
                # a list of random images for test
                # in the order of [object_amplitude, object_phase, probe_amplitude, probe_phase]
                images = [np.random.random((128,128)) for _ in range(4)]
                self.reconStepWindow.update_images(it, images)
                self.reconStepWindow.update_metric(it, data)


    def loadProbe(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open probe file', directory=self.param.working_directory, filter="(*.npy)")
        if filename is not None and len(filename) > 0:
            prb_filename = os.path.basename(filename)
            prb_dir = filename[:(len(filename)-len(prb_filename))]
            self.param.set_prb_path(prb_dir, prb_filename)
            self.le_prb_path.setText(prb_filename)
            self.ck_init_prb_flag.setChecked(False)


    def resetProbeFlg(self):
        # called when "estimate from data" is clicked
        self.param.set_prb_path('', '')
        self.le_prb_path.setText('')
        self.ck_init_prb_flag.setChecked(True)


    def loadObject(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open object file', directory=self.param.working_directory, filter="(*.npy)")
        if filename is not None and len(filename) > 0:
            obj_filename = os.path.basename(filename)
            obj_dir = filename[:(len(filename)-len(obj_filename))]
            self.param.set_obj_path(obj_dir, obj_filename)
            self.le_obj_path.setText(obj_filename)
            self.ck_init_obj_flag.setChecked(False)


    def resetObjectFlg(self):
        # called when "random start" is clicked
        self.param.set_obj_path('', '')
        self.le_obj_path.setText('')
        self.ck_init_obj_flag.setChecked(True)


    def setWorkingDirectory(self):
        dirname = QFileDialog.getExistingDirectory(self, 'Choose working folder', directory=os.path.expanduser("~"))
        if dirname is not None and len(dirname) > 0:
            dirname = dirname + "/"
            self.param.set_working_directory(dirname)
            self.le_working_directory.setText(dirname)


    def updateExtraScansFlg(self):
        self.btn_set_extra_scans.setEnabled(self.ck_extra_scans_flag.isChecked())


    def setExtraScans(self):
        if self._extra_scans_dialog is None:
            self._extra_scans_dialog = ListWidget()
            self._extra_scans_dialog.setWindowTitle('Set associated scan numbers')
            # read from param if there are any asso scans leftover from last time
            p = self.param
            if len(p.asso_scan_numbers) > 0:
                scans = self._extra_scans_dialog.listWidget
                scans.addItems([str(item) for item in p.asso_scan_numbers])
        self._extra_scans_dialog.show()
            

    def modeMultiSliceGuard(self):
        '''
        Currently our ptycho code does not support simultaneous mode + multi-slice reconstruction.
        This function can be removed once the support is added.
        '''
        if self.ck_mode_flag.isChecked() and self.ck_multislice_flag.isChecked():
           message = "Currently our ptycho code does not support simultaneous multi-mode + multi-slice reconstruction."
           print("[WARNING] " + message, file=sys.stderr)
           QtWidgets.QMessageBox.warning(self, "Warning", message)
           self.ck_mode_flag.setChecked(False)
           self.ck_multislice_flag.setChecked(False)
        self.updateModeFlg()
        self.updateMultiSliceFlg()


    def updateModeFlg(self):
        mode_flag = self.ck_mode_flag.isChecked()
        self.sp_prb_mode_num.setEnabled(mode_flag)
        self.sp_obj_mode_num.setEnabled(mode_flag)
        self.param.mode_flag = mode_flag


    def updateMultiSliceFlg(self):
        flag = self.ck_multislice_flag.isChecked()
        self.sp_slice_num.setEnabled(flag)
        self.sp_slice_spacing_m.setEnabled(flag)
        self.param.multislice_flag = flag


    def updateObjMaskFlg(self):
        mask_flag = self.ck_mask_obj_flag.isChecked()
        self.sp_amp_min.setEnabled(mask_flag)
        self.sp_amp_max.setEnabled(mask_flag)
        self.sp_pha_min.setEnabled(mask_flag)
        self.sp_pha_max.setEnabled(mask_flag)
        self.param.mask_obj_flag = mask_flag


    def checkGpuAvail(self):
        try:
            import cupy
        except ImportError:
            print('[!] Unable to import CuPy. GPU reconstruction is disabled.')
            print('[!] (Either CuPy is not installed, or GPU is not available.)')
            self.ck_gpu_flag.setChecked(False)
            self.ck_gpu_flag.setEnabled(False)
            self.param.gpu_flag = False
            self.le_gpus.setText('')
            self.le_gpus.setEnabled(False)
            self.cb_gpu_batch_size.setEnabled(False)
        else:
            del cupy


    def updateGpuFlg(self):
        flag = self.ck_gpu_flag.isChecked()
        self.le_gpus.setEnabled(flag)
        self.rb_nccl.setEnabled(flag)
        if not flag and self.rb_nccl.isChecked():
            self.rb_mpi.setChecked(True)


    def updateBraggFlg(self):
        flag = self.ck_bragg_flag.isChecked()
        self.sp_bragg_theta.setEnabled(flag)
        self.sp_bragg_gamma.setEnabled(flag)
        self.sp_bragg_delta.setEnabled(flag)
        self.param.bragg_flag = flag


    def updatePcFlg(self):
        flag = self.ck_pc_flag.isChecked()
        self.sp_pc_sigma.setEnabled(flag)
        self.sp_pc_kernel_n.setEnabled(flag)
        self.cb_pc_alg.setEnabled(flag)
        self.param.pc_flag = flag


    def updateCorrFlg(self):
        flag = self.ck_position_correction_flag.isChecked()
        self.sp_position_correction_start.setEnabled(flag)
        self.sp_position_correction_step.setEnabled(flag)
        self.param.position_correction_flag = flag


    def updateRefineDataFlg(self):
        flag = self.ck_refine_data_flag.isChecked()
        self.sp_refine_data_start_it.setEnabled(flag)
        self.sp_refine_data_interval.setEnabled(flag)
        self.sp_refine_data_step.setEnabled(flag)
        self.param.refine_data_flag = flag


    def updateBatchCropDataFlg(self):
        if self.cb_dataloader.currentText() != "Load from databroker":
            flag = False
            self.ck_batch_crop_flag.setChecked(flag)
            self.ck_batch_crop_flag.setEnabled(flag)
        else:
            flag = self.ck_batch_crop_flag.isChecked()
            self.ck_batch_crop_flag.setEnabled(True)
        self.sp_batch_x0.setEnabled(flag)
        self.sp_batch_y0.setEnabled(flag)
        self.sp_batch_width.setEnabled(flag)
        self.sp_batch_height.setEnabled(flag)


    def showNoPostProcessingWarning(self):
        if not self.ck_postprocessing_flag.isChecked():
            print("[WARNING] Post-processing is turned off. No result will be written to disk!", file=sys.stderr)


    def clearSharedMemory(self):
        message = "Are you sure you want to clear the shared memory segments currently left in /dev/shm? "\
                  "The safest way to do so is to ensure you have only one window (that is, this one) opened on this machine."
        ans = QtWidgets.QMessageBox.question(self, "Warning", message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if ans == QtWidgets.QMessageBox.Yes:
            clean_shared_memory()


    def setMPIfile(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open MPI machine file', directory=self.param.working_directory)
        if filename is not None and len(filename) > 0:
            mpi_filename = os.path.basename(filename)
            mpi_dir = filename[:(len(filename)-len(mpi_filename))]
            #self.param.le_MPI_file_path(mpi_dir, mpi_filename)
            self.param.mpi_file_path = filename
            #print(filename)
            self.le_MPI_file_path.setText(mpi_filename)
            self.le_gpus.setText('')


    def resetMPIFlg(self):
        # called when any gpu button is clicked
        self.param.mpi_file_path = ''
        self.le_MPI_file_path.setText('')


    def batchStart(self):
        if not self.ck_batch_crop_flag.isChecked() and not self.ck_batch_run_flag.isChecked():
            print("[WARNING] Choose least one action (Crop or Run). Stop.", file=sys.stderr)
            return

        if self.cb_dataloader.currentText() == "Load from databroker":
            if not self.ck_batch_crop_flag.isChecked():
                print("[WARNING] Batch mode with databroker is set, but \"Crop data\" is not.\n"
                      "[WARNING] Will attempt to load h5 from working directory", file=sys.stderr)
        
        try:
            self._scan_numbers = parse_range(self.le_batch_items.text(), self.sp_batch_step.value())
            print(self._scan_numbers)
            # TODO: is there a way to lock all widgets to prevent accidental parameter changes in the middle?

            # fire up
            self.le_scan_num.textChanged.disconnect(self.forceLoad)
            if self.ck_init_prb_batch_flag.isChecked():
                filename = self.le_prb_path_batch.text()
                self._batch_prb_filename = filename.split("*")
            if self.ck_init_obj_batch_flag.isChecked():
                filename = self.le_obj_path_batch.text()
                self._batch_obj_filename = filename.split("*")
            self._batch_manager() # serve as linked list's head
        except Exception as ex:
            self.exception_handler(ex)


    def batchStop(self):
        '''
        Brute-force abortion of the entire batch. No resumption is possible.
        '''
        self._scan_numbers = None
        self.le_scan_num.textChanged.connect(self.forceLoad)
        self.stop(True)
        if self.roiWindow is not None:
            if self.roiWindow._worker_thread is not None:
                self.roiWindow._worker_thread.disconnect()
                ## thread.terminate() freezes the whole GUI -- why?
                #if self.roiWindow._worker_thread.isRunning():
                #    self.roiWindow._worker_thread.terminate()
                #    self.roiWindow._worker_thread.wait()
                self.roiWindow._worker_thread = None
            self.roiWindow = None
        self.resetButtons()


    def _batch_manager(self):
        '''
        This is a "linked list" that utilizes Qt's signal mechanism to retrieve the next item in the list
        when the current item is processed. We need this because most likely the users want to put all
        available computing resources to process the batch item by item, and having more than one worker
        is not helping.
        '''
        # TODO: think what if anything goes wrong in the middle. Is this robust?
        if self._scan_numbers is None:
            return

        if len(self._scan_numbers) > 0:
            scan_num = self._scan_numbers.pop()
            print("[BATCH] begin processing scan " + str(scan_num) + "...")
            self.le_scan_num.setText(str(scan_num))
            self.btn_recon_batch_start.setEnabled(False)
            self.btn_recon_batch_stop.setEnabled(True)

            if self.ck_batch_crop_flag.isChecked():
                self._batch_crop()  # also handles "Run" if needed
            elif self.ck_batch_run_flag.isChecked():
                self._batch_run()  # h5 exists, just "Run"
            else:
                raise
        else:
            print("[BATCH] batch processing complete!")
            self._scan_numbers = None
            self.le_scan_num.textChanged.connect(self.forceLoad)
            self.resetButtons()
            if self.roiWindow is not None:
                self.roiWindow = None


    def _batch_crop(self):
        # ugly hack: pretend the ROI window exists, take the first frame for finding bad pixels,
        # mimic human input, and run the reconstruction (if checked)

        # first get params from databroker
        eventloop = QtCore.QEventLoop()
        self._mainwindow_signal.connect(eventloop.quit)
        self.loadExpParam()
        eventloop.exec()

        # then invoke the h5 worker in RoiWindow
        if self.roiWindow is not None:
            self.roiWindow.close()
        img = self._viewDataFrameBroker(0)
        self.roiWindow = RoiWindow(image=img, main_window=self)
        #self.roiWindow.roi_changed.connect(self._get_roi_slot)
        self.roiWindow.canvas._eventHandler.set_curr_roi(self.roiWindow.canvas.ax,
            (self.sp_batch_x0.value(), self.sp_batch_y0.value()),
            self.sp_batch_width.value(), self.sp_batch_height.value())
        #print("ROI:", self.roiWindow.canvas.get_red_roi())
        self.roiWindow.save_to_h5()
        #self.btn_recon_batch_stop.clicked.connect(self.roiWindow._worker_thread.terminate)
        if not self.ck_batch_run_flag.isChecked():
            self.roiWindow._worker_thread.finished.connect(self._batch_manager)
        else:
            self.roiWindow._worker_thread.finished.connect(self._batch_run)


    def _batch_run(self):
        self.loadExpParam()
        self.start(True)


    def switchProbeBatch(self):
        if self.ck_init_prb_batch_flag.isChecked():
            self.le_prb_path_batch.setEnabled(True)
        else:
            self.le_prb_path_batch.setEnabled(False)
            self.le_prb_path_batch.setText('')
            self._batch_prb_filename = None


    def switchObjectBatch(self):
        if self.ck_init_obj_batch_flag.isChecked():
            self.le_obj_path_batch.setEnabled(True)
        else:
            self.le_obj_path_batch.setEnabled(False)
            self.le_obj_path_batch.setText('')
            self._batch_obj_filename = None


    def viewDataFrame(self):
        '''
        Correspond to "View & set" in DPC GUI
        '''
        if _TEST:
            image = load_image_pil('./test.tif')
            self.roiWindow = RoiWindow(image=image)
            self.roiWindow.roi_changed.connect(self._get_roi_slot)
            self.roiWindow.show()
            return

        if not self._loaded:
            print("[WARNING] Remember to click \"Load\" before proceeding!", file=sys.stderr)
            return

        frame_num = self.sp_fram_num.value()
        img = None

        try:
            if self.cb_dataloader.currentText() == "Load from databroker":
                img = self._viewDataFrameBroker(frame_num)
            
            if self.cb_dataloader.currentText() == "Load from h5":
                img = self._viewDataFrameH5(frame_num)
        except OSError:
            # h5 not found, but loadExpParam() has detected it, so do nothing here
            pass
        except (ValueError, RuntimeError) as ex:
            # let user follow the instruction to make correction
            print(ex, file=sys.stderr)
        except Exception as ex:
            # don't expect this will happen but if so I'd like to know what
            self.exception_handler(ex)
        else:
            if self.roiWindow is None:
                self.roiWindow = RoiWindow(image=img, main_window=self)
            #else:
            #    self.roiWindow.reset_window(image=img, main_window=self)
            ##self.roiWindow.roi_changed.connect(self._get_roi_slot)
            self.roiWindow.show()


    #@profile
    def _viewDataFrameBroker(self, frame_num:int):
        # assuming at this point the user has clicked "load" 
        if not self._loaded:
            raise RuntimeError("[ERROR] Need to click the \"load\" button before viewing.")
        if self._mds_table is not None:
            return get_single_image(self._db, frame_num, self._mds_table)
        else:
            scan_num = int(self.le_scan_num.text())
            items = []
            if self._extra_scans_dialog is not None:
                list_widget = self._extra_scans_dialog.listWidget
                for idx in range(list_widget.count()):
                    items.append(int(list_widget.item(idx).text()))
    #                items.append(list_widget.item(idx).data(QtCore.Qt.UserRole))#.toPyObject())
    #            items = [item.text() for item in list_widget.items()]
    #        print(items, self._db)
            return get_single_image(self._db, frame_num, scan_num, *items)


    #@profile
    def _viewDataFrameH5(self, frame_num:int):
        # load the data from the h5 in the working directory
        working_dir = str(self.le_working_directory.text()) # self.param.working_directory
        scan_num = str(self.le_scan_num.text())
        length = self.sp_num_points.value()
        if frame_num >= length:
            message = "[ERROR] The {0}-th frame doesn't exist. "
            message += "Available frames for the chosen scan: [0, {1}]."
            raise ValueError(message.format(frame_num, length-1))
        with h5py.File(working_dir+'/scan_'+scan_num+'.h5','r') as f:
            print("h5 loaded, parsing the {}-th frame...".format(frame_num), end='')
            img = f['diffamp'][frame_num]
            #data = f['diffamp'].value
            #img = data[frame_num]
            print("done")
        return img


    def _get_roi_slot(self, x0, y0, width, height):
        '''
        feel free to rename this function as you need
        : this function to get roi when user click SEND button or
        : dynamically...

        x0: upper left x coordinate
        y0: upper left y coordinate
        width: width
        height: height
        '''
        print(x0, y0, width, height)


    def loadExpParam(self):
        scan_num = self.le_scan_num.text()

        try:
            if self.cb_dataloader.currentText() == "Load from databroker":
                self._loadExpParamBroker(int(scan_num))

            if self.cb_dataloader.currentText() == "Load from h5":
                self._loadExpParamH5(scan_num)
        except OSError: # for h5
            print("[ERROR] h5 not found. Resetting...", file=sys.stderr, end='')
            self.resetExperimentalParameters()
        except Exception as ex: # everything unexpected at this time...
            self.exception_handler(ex)
        else:
            self._loaded = True


    #@profile
    def _loadExpParamBroker(self, scan_id:int):
        self.db = scan_id # set the correct database
        header = self.db[scan_id]

        # get the list of detector names
        det_names = get_detector_names(self.db, scan_id)
        det_name = self.cb_detectorkind.currentText()
        det_name_exists = False
        self.cb_detectorkind.clear()
        for detector_name in det_names:
            self.cb_detectorkind.addItem(detector_name)
            if det_name == detector_name:
                det_name_exists = True
        if not det_name_exists:
            det_name = self.cb_detectorkind.currentText()

        # get metadata
        thread = self._worker_thread \
               = HardWorker("fetch_data", self.db, scan_id, det_name)
        thread.update_signal.connect(self._setExpParamBroker)
        thread.finished.connect(lambda: self.btn_load_scan.setEnabled(True))
        thread.exception_handler = self.exception_handler
        self.btn_load_scan.setEnabled(False)
        thread.start()


    def _setExpParamBroker(self, it, metadata:dict):
        '''
        Notes:
        1. The parameter "it" is just a placeholder for the signal 
        2. The exceptions are handled in the HardWorker thread, so this function
           is guaranteed no-throw.
        '''
        #metadata = load_metadata(self.db, scan_id, det_name)
        self.param.__dict__ = {**self.param.__dict__, **metadata} # for Python 3.5+ only

        # get the mds keys to the image (diffamp) array 
        self._mds_table = metadata.get('mds_table')

        # update experimental parameters
        self.sp_xray_energy.setValue(metadata['xray_energy_kev'])
        if 'z_m' in metadata:
            print("[WARNING] Retrieved and updated the detector distance (from a hard-coded source).", file=sys.stderr)
            self.sp_detector_distance.setValue(metadata['z_m'])
        self.sp_x_arr_size.setValue(metadata['nx'])
        self.sp_y_arr_size.setValue(metadata['ny'])
        self.sp_num_points.setValue(metadata['nz'])
        self.sp_x_step_size.setValue(metadata['dr_x'])
        self.sp_y_step_size.setValue(metadata['dr_y'])
        self.sp_x_scan_range.setValue(metadata['x_range'])
        self.sp_y_scan_range.setValue(metadata['y_range'])
        self.sp_ccd_pixel_um.setValue(metadata['ccd_pixel_um'])
        self.sp_angle.setValue(metadata['angle'])
        if self.cb_scan_type.findText(metadata['scan_type']) == -1:
            self.cb_scan_type.addItem(metadata['scan_type'])
        self.cb_scan_type.setCurrentText(metadata['scan_type'])
        self._scan_points = metadata['points']
        print("done")
        self._mainwindow_signal.emit()


    def setLoadButton(self):
        if self.cb_dataloader.currentText() == "Load from databroker":
            self.cb_detectorkind.setEnabled(True)
            self.cb_scan_type.setEnabled(True)
            if beamline_name == 'HXN':
                #print("[WARNING] Currently detector distance is unavailable in Databroker and must be set manually!", file=sys.stderr)
                print("[WARNING] Detector distance is unavailable in Databroker, assumed to be 0.5m", file=sys.stderr)
                self.sp_detector_distance.setValue(0.5)
        if self.cb_dataloader.currentText() == "Load from h5":
            self.cb_detectorkind.setEnabled(False)
            self.cb_scan_type.setEnabled(False) # do we ever write scan type to h5???


    #@profile
    def _loadExpParamH5(self, scan_num:str):
        # load the parameters from the h5 in the working directory
        working_dir = str(self.le_working_directory.text()) # self.param.working_directory
        with h5py.File(working_dir+'/scan_'+scan_num+'.h5','r') as f:
            # this code is not robust enough as certain keys may not be present...
            print("h5 loaded, parsing experimental parameters...", end='')
            self.sp_xray_energy.setValue(1.2398/f['lambda_nm'][()])
            self.sp_detector_distance.setValue(f['z_m'][()])
            nz, nx, ny = f['diffamp'].shape
            self.sp_x_arr_size.setValue(nx)
            self.sp_y_arr_size.setValue(ny)
            self.sp_num_points.setValue(nz)
            self.sp_x_step_size.setValue(f['dr_x'][()])
            self.sp_y_step_size.setValue(f['dr_y'][()])
            self.sp_x_scan_range.setValue(f['x_range'][()])
            self.sp_y_scan_range.setValue(f['y_range'][()])
            self.sp_ccd_pixel_um.setValue(f['ccd_pixel_um'][()])
            if 'angle' in f.keys():
                self.sp_angle.setValue(f['angle'][()])
            else:
                self.sp_angle.setValue(15.) # backward compatibility for old datasets
                print("[WARNING] angle not found, assuming 15...", file=sys.stderr)
            self._scan_points = f['points'][:] # for visualization purpose
            #self.cb_scan_type = ...
            # read the detector name and set it in GUI??
            print("done")


    def importConfig(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Select GUI config file', directory=self.param.working_directory, filter="(*.txt)")
        if filename is not None and len(filename) > 0:
            try:
                self.param = parse_config(filename, self.param)

                ## update exp parameters since this is supposed to be handled by "Load"
                #self.sp_xray_energy.setValue(p.xray_energy_kev)
                #self.sp_detector_distance.setValue(p.z_m)
                #self.sp_x_arr_size.setValue(p.nx)
                #self.sp_y_arr_size.setValue(p.ny)
                #self.sp_x_step_size.setValue(p.dr_x)
                #self.sp_y_step_size.setValue(p.dr_y)
                #self.sp_x_scan_range.setValue(p.x_range)
                #self.sp_y_scan_range.setValue(p.y_range)
                #self.sp_angle.setValue(p.angle)
                ##self.cb_scan_type = ...
                #self.sp_num_points.setValue(p.nz)

                self.update_gui_from_param()
            except Exception as ex:
                self.exception_handler(ex)
            else:
                print("config loaded from " + filename)
                self._loaded = True
                

    def exportConfig(self):
        self.update_param_from_gui()
        filename, _ = QFileDialog.getSaveFileName(self, 'Save GUI config to txt', directory=self.param.working_directory, filter="(*.txt)")
        if filename is not None and len(filename) > 0:
            if filename[-4:] != ".txt":
                filename += ".txt"
            self._exportConfigHelper(filename)
            print("config saved to " + filename)


    def _exportConfigHelper(self, filename:str):
        keys = list(self.param.__dict__.keys())
        keys.sort()
        with open(filename, 'w') as f:
            f.write("[GUI]\n")
            for key in keys:
                # skip a few items related to databroker
                if key == 'points' or key == 'ic' or key == 'mds_table':
                    continue
                f.write(key+" = "+str(self.param.__dict__[key])+"\n")


    def resetExperimentalParameters(self):
        self.sp_xray_energy.setValue(0)
        self.sp_detector_distance.setValue(0)
        self.sp_x_arr_size.setValue(0)
        self.sp_y_arr_size.setValue(0)
        self.sp_x_step_size.setValue(0)
        self.sp_y_step_size.setValue(0)
        self.sp_x_scan_range.setValue(0)
        self.sp_y_scan_range.setValue(0)
        #self.cb_scan_type = ...
        self.sp_num_points.setValue(0)

    
    def forceLoad(self):
        '''
        A foolproof mechanism that forces users to click "load" before "start" or "view data frame".
        This can avoid handling many weird exceptions.
        '''
        self._loaded = False

        # if there's a roiWindow, we should close it and reopen to flush out old data
        if self.roiWindow is not None:
            self.roiWindow.close()
            self.roiWindow = None


    def exception_handler(self, ex):
        formatted_lines = traceback.format_exc().splitlines()
        for line in formatted_lines:
            print("[ERROR] " + line, file=sys.stderr) 
        print("[ERROR] " + str(ex), file=sys.stderr)


    def closeEvent(self, event):
        # Overwrite the class's default
        message = "Are you sure you want to quit the app?"
        ans = QtWidgets.QMessageBox.question(self, "Warning", message, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if ans == QtWidgets.QMessageBox.Yes:
            self.stop()
            if self.reconStepWindow is not None:
                self.reconStepWindow.close()
            if self.roiWindow is not None:
                self.roiWindow.close()
            if self.scanWindow is not None:
                self.scanWindow.close()
            if self._extra_scans_dialog is not None:
                self._extra_scans_dialog.close()
            event.accept()
        else:
            event.ignore()


    # reimplementing __del__ is useless, so use the signal QApplication.aboutToQuit
    def destructor(self):
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            if self.menu_save_config_history.isChecked():
                self.update_param_from_gui()
                self._exportConfigHelper(self._config_path)
                #print("config file was written to " + self._config_path)
            self.close_mmap()
        except:
            traceback.print_exc()
            raise


    @QtCore.pyqtSlot(str, QtGui.QColor)
    def on_stdout_message(self, message, color):
        self.console_info.moveCursor(QtGui.QTextCursor.End)
        self.console_info.setTextColor(color)
        self.console_info.insertPlainText(message)



def main():
    app = QtWidgets.QApplication(sys.argv)

    w = MainWindow()
    w.show()
    app.installEventFilter(w)

    console_stdout = PtychoStream(color = "black")
    console_stderr = PtychoStream(color = "red")
    console_stdout.message.connect(w.on_stdout_message)
    console_stderr.message.connect(w.on_stdout_message)

    app.aboutToQuit.connect(w.destructor)

    sys.stdout = console_stdout
    sys.stderr = console_stderr
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
