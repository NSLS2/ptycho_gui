from PyQt5 import QtCore
from datetime import datetime
from .ptycho_param import Param
import sys, os
import pickle     # dump param into disk
import subprocess # call mpirun from shell
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
import numpy as np
import traceback
import time


import requests
import json

from .databroker_api import load_metadata, save_data
from .utils import use_mpi_machinefile, set_flush_early
from .ptycho.utils import save_config


class RemoteJobHandler:
    def __init__(self):
        self.url =  "https://orion-api-staging.nsls2.bnl.gov/api/v1/compute/orion/jobs"
        self.api_key = os.getenv("APIKEY")
        self.headers =  {
            "Content-Type": "application/json",
            "x-api-key": f"{self.api_key}"
            }
        self.params = {"expand_info": False}

    def submit_job(self, remote_path, param):
        print("inside submit job with path ", remote_path)
        print("inside submit job num_gpus ", param.gpus)

        #remote_path =  "/nsls2/users/skarakuzu1/orion_ptycho"

        srun_command = "python -W ignore -m nsls2ptycho.core.ptycho.recon_ptycho_gui /nsls2/users/skarakuzu1/ptycho_test/remote_orion/ptycho_320045_t1"
        #srun_command = "python solve.py"

        overrides = {
            "name": "trial",
            "partition": "normal",
            "tasks": f"{len(param.gpus)}",
            "time_limit": 720,
            "tres_per_task": "cpu=1,gres/gpu=1",
            "standard_output": "trial.out",
            "standard_error": "trial.err",
        }

        payload = {
            "script": (
                "#!/bin/bash -l\n"
                "module load orion/gpu\n"
                "module unload openmpi\n"
                "conda activate /nsls2/conda/envs/2025-2.0-py311-tiled/\n"
                "nvidia-smi\n"
                "echo $(pwd)\n"
                "echo $(which mpicc)\n"
                #f"mpirun -n 2 {srun_command}\n"
                f"srun --mpi=pmix {srun_command}\n"
            ),
            "working_dir_path": f"{remote_path}",
            "environment": [
                "PATH=/usr/bin:/bin:/usr/sbin:/sbin",
                "HOME=/nsls2/users/skarakuzu1",
                "SLURM_EXPORT_ENV=ALL",
            ],
            "overrides": overrides,
        }


        print("printing payload")
        print(payload)
        sys.stdout.flush()

        response = requests.post(self.url, headers=self.headers, json=payload)
        resp_json = response.json()
        print("response is ", response.status_code, resp_json)
        
        if response.status_code == 200:
            self.remote_job_id = resp_json['job_id']
            print("job_id is ", self.remote_job_id)


        return response.status_code


    def cancel_job(self):
        response = requests.delete(f"{self.url}/{self.remote_job_id}", headers=self.headers, params=self.params)

        resp_json = response.json()
        print("response is ", response.status_code, resp_json)

        return response.status_code

    def get_job_status(self):
        response = requests.get(f"{self.url}/{self.remote_job_id}", headers=self.headers, params=self.params)

        resp_json = response.json()
        print("response is ", response.status_code, resp_json)

        if response.status_code == 200:
            state = resp_json["jobs"][0]["state"][0]
        return state


class PtychoReconRemote(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(int, object) # (interation number, chi arrays)

    def __init__(self, param:Param=None, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.param = param

        self.return_value = None

        self.remote_path = os.path.join(os.path.realpath(self.param.working_directory),'remote_'+self.param.remote_srv)
        if not os.path.isdir(self.remote_path):
            os.mkdir(self.remote_path)

        self.msg_file = os.path.join(os.path.join(self.remote_path,'msg'))
        if not os.path.isfile(self.msg_file):
            with open(self.msg_file,'w') as f:
                pass
        self.msg = open(self.msg_file,'r')
        self.msg.readlines()
         
        self.remote_job_handler = RemoteJobHandler()

    def _parse_message(self, tokens):
        def _parser(current, upper_limit, target_list):
            for j in range(upper_limit):
                target_list.append(float(tokens[current+2+j]))
    
        # assuming tokens (stdout line) is split but not yet processed
        it = int(tokens[2])
        
        # first remove brackets
        empty_index_list = []
        for i, token in enumerate(tokens):
            tokens[i] = token.replace('[', '').replace(']', '')
            if tokens[i] == '':
                empty_index_list.append(i)
        counter = 0
        for i in empty_index_list:
            del tokens[i-counter]
            counter += 1

        # next parse based on param and the known format
        prb_list = []
        obj_list = []
        for i, token in enumerate(tokens):
            if token == 'probe_chi':
                _parser(i, self.param.prb_mode_num, prb_list)
                #elif self.param.multislice_flag: 
            if token == 'object_chi':
                if not self.param.multislice_flag:
                    _parser(i, self.param.obj_mode_num, obj_list)
                else:
                    _parser(i, self.param.slice_num, obj_list)

        # return a dictionary
        result = {'probe_chi':prb_list, 'object_chi':obj_list}

        return it, result

    def _test_stdout_completeness(self, stdout):
        counter = 0
        for token in stdout:
            if token == '=':
                counter += 1

        return counter

    def _parse_one_line(self):
        stdout_2 = self.process.stdout.readline().decode('utf-8')
        print(stdout_2, end='') # because the line already ends with '\n'

        return stdout_2.split()

    def export_slurm_header(self):
        slurm_header = os.path.expanduser("~") + "/.ptycho_gui/.ptycho_slurm"
        with open(slurm_header, 'w') as f:
            f.write(self.remote_path+' '+str(len(self.param.gpus))+'\n')

    def clear_slurm_header(self):
        slurm_header = os.path.expanduser("~") + "/.ptycho_gui/.ptycho_slurm"
        if os.path.exists(slurm_header):
            try:
                os.remove(slurm_header)
            except:
                pass
   

    def recon_remote(self, param:Param, update_fcn=None):

        self.fname_full = os.path.join(self.remote_path,'ptycho_'+str(param.scan_num)+'_'+param.sign)
        
        if param.working_directory:
            param.working_directory = os.path.realpath(param.working_directory)+'/'
        if param.prb_dir:
            param.prb_dir = os.path.realpath(param.prb_dir)+'/'
        if param.prb_path:
            param.prb_path = os.path.realpath(param.prb_path)
        if param.obj_dir:
            param.obj_dir = os.path.realpath(param.obj_dir)+'/'
        if param.obj_path:
            param.obj_path = os.path.realpath(param.obj_path)
        
        save_config(self.fname_full,param)
        self.export_slurm_header()

        self.return_value = 0 # Assume the recon will succeed unless later detects failure and modify it.

        #bearer_token = os.getenv("SLURM_JWT")
        status = self.remote_job_handler.submit_job(self.remote_path, param)
        print("Submitted job from the gui")


        # try:
        #time.sleep(1)
        #while not out:
        #    print('Waiting for remote worker on %s to take the recon task...'%param.remote_srv)
        #    time.sleep(1)
        #    out = self.msg.readlines()
        #    if os.path.isfile(os.path.join(self.remote_path,'abort')):
        #        os.remove(os.path.join(self.remote_path,'abort'))
        #        if os.path.isfile(os.path.join(self.remote_path,'msg')):
        #            os.remove(os.path.join(self.remote_path,'msg'))
        #        if os.path.isfile(self.fname_full):
        #            os.remove(self.fname_full)
        #        raise Exception('Remote recon aborted...')

        while self.remote_job_handler.get_job_status() != "RUNNING":
            print('Waiting for remote worker on %s to take the recon task...'%param.remote_srv)


        time.sleep(1)
        print("DEBUG: Attempting to read job output...")

        file_name = f"slurm-{self.remote_job_handler.remote_job_id}.out"
        msg_file = os.path.join(self.remote_path, file_name)
        
        print("DEBUG: msg_file =", msg_file)
        print("DEBUG: exists? ", os.path.exists(msg_file))

        try:
            with open(msg_file, "r") as f:
                print("DEBUG: opened successfully")
                print(f.read())
        except Exception as e:
            print("DEBUG: ERROR opening file:", e)

        msg = open(msg_file, "r") 
        out = None
        while not out:
            print('Waiting for remote worker on %s to take the recon task...'%param.remote_srv)
            time.sleep(1)
            out = msg.readlines()
            #if os.path.isfile(os.path.join(self.remote_path,'abort')):
            #    os.remove(os.path.join(self.remote_path,'abort'))
            #    if os.path.isfile(os.path.join(self.remote_path,'msg')):
            #        os.remove(os.path.join(self.remote_path,'msg'))
            #    if os.path.isfile(self.fname_full):
            #        os.remove(self.fname_full)
            #    raise Exception('Remote recon aborted...')

        

        while True:
            for line in out:
                print(line, end='') # because the line already ends with '\n'
                tokens = line.split()
                if len(tokens) > 2 and tokens[0] == "[INFO]" and update_fcn is not None:
                    it, result = self._parse_message(tokens)
                    self.parent.it_last = it
                    update_fcn(it+1, result)
                    #print(result['probe_chi'])
                if 'aborted' in line:
                    self.return_value = 1 # Aborted
            if not os.path.isfile(self.fname_full):
                break
            
            time.sleep(0.1)
            out = msg.readlines()
            #out = self.msg.readlines()
        # except:
        #     pass
        # finally:
        #     pass

    def run(self):
        print('Ptycho thread started helloooo***')
        try:
            self.recon_remote(self.param, self.update_signal.emit)
        except IndexError:
            print("[ERROR] IndexError --- most likely a wrong MPI machine file is given?", file=sys.stderr)
        except:
            # whatever happened in the MPI processes will always (!) generate traceback,
            # so do nothing here
            pass
        else:
            # let preview window load results
            if self.param.preview_flag and self.return_value==0:
                self.update_signal.emit(self.param.n_iterations+1,None)
            
        finally:
            self.clear_slurm_header()
            print("Cancelling from the gui")
            self.remote_job_handler.cancel_job()
            print('finally?')

    def kill(self):
        print("In the kill section")
        if os.path.isdir(self.remote_path):
            with open(os.path.join(self.remote_path,'abort'),'w') as f:
                pass

class PtychoReconWorker(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(int, object) # (interation number, chi arrays)
    process = None # subprocess 

    def __init__(self, param:Param=None, parent=None):
        super().__init__(parent)
        self.param = param
        self.return_value = None

    def _parse_message(self, tokens):
        def _parser(current, upper_limit, target_list):
            for j in range(upper_limit):
                target_list.append(float(tokens[current+2+j]))
    
        # assuming tokens (stdout line) is split but not yet processed
        it = int(tokens[2])
        
        # first remove brackets
        empty_index_list = []
        for i, token in enumerate(tokens):
            tokens[i] = token.replace('[', '').replace(']', '')
            if tokens[i] == '':
                empty_index_list.append(i)
        counter = 0
        for i in empty_index_list:
            del tokens[i-counter]
            counter += 1

        # next parse based on param and the known format
        prb_list = []
        obj_list = []
        for i, token in enumerate(tokens):
            if token == 'probe_chi':
                _parser(i, self.param.prb_mode_num, prb_list)
            if token == 'object_chi':
                if not self.param.multislice_flag:
                    _parser(i, self.param.obj_mode_num, obj_list)
                else:
                    _parser(i, self.param.slice_num, obj_list)

        # return a dictionary
        result = {'probe_chi':prb_list, 'object_chi':obj_list}

        return it, result

    def _test_stdout_completeness(self, stdout):
        counter = 0
        for token in stdout:
            if token == '=':
                counter += 1

        return counter

    def _parse_one_line(self):
        stdout_2 = self.process.stdout.readline().decode('utf-8')
        print(stdout_2, end='') # because the line already ends with '\n'

        return stdout_2.split()

    def recon_api(self, param:Param, update_fcn=None):
        parent_module = '.'.join(self.__module__.rsplit('.', 2)[:-1]) # get parent module name to run the correct recon worker
        # "1" is just a placeholder to be overwritten soon
        mpirun_command = ["mpirun", "-n", "1", "python", "-W", "ignore", "-m",parent_module+".ptycho.recon_ptycho_gui"]

        if param.mpi_file_path == '':
            if param.gpu_flag:
                mpirun_command[2] = str(len(param.gpus))
            else:
                mpirun_command[2] = str(param.processes) if param.processes > 1 else str(1)
        else:
            # regardless if GPU is used or not --- trust users to know this
            mpirun_command = use_mpi_machinefile(mpirun_command, param.mpi_file_path)

        mpirun_command = set_flush_early(mpirun_command)

        # for CuPy v8.0+
        os.environ['CUPY_ACCELERATORS'] = 'cub'
                
        try:
            self.return_value = None
            with subprocess.Popen(mpirun_command,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  env=dict(os.environ, mpi_warn_on_fork='0')) as run_ptycho:
                self.process = run_ptycho # register the subprocess

                # idea: if we attempts to readline from an empty pipe, it will block until 
                # at least one line is piped in. However, stderr is ususally empty, so reading
                # from it is very likely to block the output until the subprocess ends, which 
                # is bad. Thus, we want to set the O_NONBLOCK flag for stderr, see
                # http://eyalarubas.com/python-subproc-nonblock.html 
                #
                # Note that it is unclear if readline in Python 3.5+ is guaranteed safe with 
                # non-blocking pipes or not. See https://bugs.python.org/issue1175#msg56041 
                # and https://stackoverflow.com/questions/375427/
                # If this is a concern, using the asyncio module could be a safer approach?
                # One could also process stdout in one loop and then stderr in another, which
                # will not have the blocking issue.
                flags = fcntl(run_ptycho.stderr, F_GETFL) # first get current stderr flags
                fcntl(run_ptycho.stderr, F_SETFL, flags | O_NONBLOCK)

                while True:
                    stdout = run_ptycho.stdout.readline()
                    stderr = run_ptycho.stderr.readline() # without O_NONBLOCK this will very likely block
                    
                    if (run_ptycho.poll() is not None) and (stdout==b'') and (stderr==b''):
                        break

                    if stdout:
                        stdout = stdout.decode('utf-8')
                        print(stdout, end='') # because the line already ends with '\n'
                        stdout = stdout.split()
                        if len(stdout) > 2 and stdout[0] == "[INFO]" and update_fcn is not None:
                            # TEST: check if stdout is complete by examining the number of "="
                            # TODO: improve this ugly hack...
                            while True:
                                counter = self._test_stdout_completeness(stdout)
                                if counter == 3:
                                    break
                                elif counter < 3:
                                    stdout += self._parse_one_line()
                                else: # counter > 3, we read one more line!
                                    raise Exception("parsing error")
                          
                            it, result = self._parse_message(stdout)
                            #print(result['probe_chi'])
                            update_fcn(it+1, result)
                        elif len(stdout) == 3 and stdout[0] == "shared" and update_fcn is not None:
                            update_fcn(-1, "init_mmap")

                    if stderr:
                        stderr = stderr.decode('utf-8')
                        print(stderr, file=sys.stderr, end='')

                # get the return value 
                self.return_value = run_ptycho.poll()

            if self.return_value != 0:
                message = "At least one MPI process returned a nonzero value, so the whole job is aborted.\n"
                message += "If you did not manually terminate it, consult the Traceback above to identify the problem."
                raise Exception(message)
        except Exception as ex:
            traceback.print_exc()
            #print(ex, file=sys.stderr)
            #raise ex
        finally:
            # clean up temp file
            filepath = param.working_directory + "/." + param.shm_name + ".txt"
            if os.path.isfile(filepath):
                os.remove(filepath)

    def run(self):
        print('Ptycho thread started')
        try:
            self.recon_api(self.param, self.update_signal.emit)
        except IndexError:
            print("[ERROR] IndexError --- most likely a wrong MPI machine file is given?", file=sys.stderr)
        except:
            # whatever happened in the MPI processes will always (!) generate traceback,
            # so do nothing here
            pass
        else:
            # let preview window load results
            if self.param.preview_flag and self.return_value == 0:
                self.update_signal.emit(self.param.n_iterations+1, None)
        finally:
            print('finally?')

    def kill(self):
        if self.process is not None:
            print('killing the subprocess...')
            self.process.terminate()
            self.process.wait()

class PtychoReconLive(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(int, object) # (interation number, chi arrays)
    process = None # subprocess 

    def __init__(self, param:Param=None, parent=None):
        super().__init__(parent)
        self.param = param
        self.config_file = parent._config_path
        self.return_value = None

    def _parse_message(self, tokens):
        def _parser(current, upper_limit, target_list):
            for j in range(upper_limit):
                target_list.append(float(tokens[current+2+j]))
    
        # assuming tokens (stdout line) is split but not yet processed
        try:
            it = int(tokens[2])
        except:
            return
        
        # first remove brackets
        empty_index_list = []
        for i, token in enumerate(tokens):
            tokens[i] = token.replace('[', '').replace(']', '')
            if tokens[i] == '':
                empty_index_list.append(i)
        counter = 0
        for i in empty_index_list:
            del tokens[i-counter]
            counter += 1

        # next parse based on param and the known format
        prb_list = []
        obj_list = []
        for i, token in enumerate(tokens):
            if token == 'probe_chi':
                _parser(i, self.param.prb_mode_num, prb_list)
            if token == 'object_chi':
                if not self.param.multislice_flag:
                    _parser(i, self.param.obj_mode_num, obj_list)
                else:
                    _parser(i, self.param.slice_num, obj_list)

        # return a dictionary
        result = {'probe_chi':prb_list, 'object_chi':obj_list}

        return it, result

    def _test_stdout_completeness(self, stdout):
        counter = 0
        for token in stdout:
            if token == '=':
                counter += 1

        return counter

    def _parse_one_line(self):
        stdout_2 = self.process.stdout.readline().decode('utf-8')
        print(stdout_2, end='') # because the line already ends with '\n'

        return stdout_2.split()

    def recon_api(self, param:Param, update_fcn=None):
        parent_module = '.'.join(self.__module__.rsplit('.', 2)[:-1]) # get parent module name to run the correct recon worker
        # "1" is just a placeholder to be overwritten soon
        if param.gpu_flag and len(param.gpus) == 1:
            mpirun_command = ["python", "-W", "ignore", "-m",parent_module+".Holoptycho",self.config_file]
        else:
            raise NotImplementedError('Live recon on multiple gpus not implemented')
        
        mpirun_command = set_flush_early(mpirun_command)

        # for CuPy v8.0+
        os.environ['CUPY_ACCELERATORS'] = 'cub'

        print(mpirun_command)
                
        try:
            self.return_value = None
            with subprocess.Popen(mpirun_command,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                  env=dict(os.environ, mpi_warn_on_fork='0')) as run_ptycho:
                self.process = run_ptycho # register the subprocess

                # idea: if we attempts to readline from an empty pipe, it will block until 
                # at least one line is piped in. However, stderr is ususally empty, so reading
                # from it is very likely to block the output until the subprocess ends, which 
                # is bad. Thus, we want to set the O_NONBLOCK flag for stderr, see
                # http://eyalarubas.com/python-subproc-nonblock.html 
                #
                # Note that it is unclear if readline in Python 3.5+ is guaranteed safe with 
                # non-blocking pipes or not. See https://bugs.python.org/issue1175#msg56041 
                # and https://stackoverflow.com/questions/375427/
                # If this is a concern, using the asyncio module could be a safer approach?
                # One could also process stdout in one loop and then stderr in another, which
                # will not have the blocking issue.
                flags = fcntl(run_ptycho.stdout, F_GETFL) # first get current stderr flags
                fcntl(run_ptycho.stdout, F_SETFL, flags | O_NONBLOCK)
                flags = fcntl(run_ptycho.stderr, F_GETFL) # first get current stderr flags
                fcntl(run_ptycho.stderr, F_SETFL, flags | O_NONBLOCK)

                while True:
                    try:
                        stdout = run_ptycho.stdout.readline()
                        stderr = run_ptycho.stderr.readline() # without O_NONBLOCK this will very likely block
                    except:
                        traceback.print_exc()
                    
                    if (run_ptycho.poll() is not None) and (stdout==b'') and (stderr==b''):
                        break

                    if stdout:
                        stdout = stdout.decode('utf-8')
                        print(stdout, end='') # because the line already ends with '\n'
                        stdout = stdout.split()
                        if len(stdout) > 2 and stdout[0] == "[INFO]" and update_fcn is not None:
                            # TEST: check if stdout is complete by examining the number of "="
                            # TODO: improve this ugly hack...
                            while True:
                                counter = self._test_stdout_completeness(stdout)
                                if counter == 3:
                                    break
                                elif counter < 3:
                                    stdout += self._parse_one_line()
                                else: # counter > 3, we read one more line!
                                    raise Exception("parsing error")
                          
                            it, result = self._parse_message(stdout)
                            #print(result['probe_chi'])
                            update_fcn(it+1, result)
                        elif len(stdout) == 3 and stdout[0] == "shared" and update_fcn is not None:
                            update_fcn(-1, "init_mmap")
                        elif len(stdout) == 3 and stdout[0] == "flush" and update_fcn is not None:
                            update_fcn(-1, "flush")
                        elif len(stdout) == 3 and stdout[0] == "reload" and update_fcn is not None:
                            update_fcn(-1, "reload")

                    if stderr:
                        stderr = stderr.decode('utf-8')
                        print(stderr, file=sys.stderr, end='')

                # get the return value 
                self.return_value = run_ptycho.poll()

            if self.return_value != 0:
                message = "At least one MPI process returned a nonzero value, so the whole job is aborted.\n"
                message += "If you did not manually terminate it, consult the Traceback above to identify the problem."
                raise Exception(message)
        except:
            traceback.print_exc()
            #print(ex, file=sys.stderr)
            #raise ex
        finally:
            pass
            # clean up temp file
            filepath = param.working_directory + "/." + param.shm_name + ".txt"
            if os.path.isfile(filepath):
                os.remove(filepath)

    def run(self):
        print('Ptycho thread started')
        try:
            self.recon_api(self.param, self.update_signal.emit)
        except IndexError:
            print("[ERROR] IndexError --- most likely a wrong MPI machine file is given?", file=sys.stderr)
        except:
            # whatever happened in the MPI processes will always (!) generate traceback,
            # so do nothing here
            pass
        else:
            # let preview window load results
            if self.param.preview_flag and self.return_value == 0:
                self.update_signal.emit(self.param.n_iterations+1, None)
        finally:
            print('finally?')

    def kill(self):
        if self.process is not None:
            print('killing the subprocess...')
            self.process.terminate()
            self.process.wait()

# a worker that does the rest of hard work for us
class HardWorker(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(int, object) # connect to MainWindow???
    def __init__(self, task=None, *args, parent=None):
        super().__init__(parent)
        self.task = task
        self.args = args
        self.exception_handler = None
        #self.update_signal = QtCore.pyqtSignal(int, object) # connect to MainWindow???

    def run(self):
        try:
            if self.task == "save_h5":
                self._save_h5(self.update_signal.emit)
            elif self.task == "fetch_data":
                self._fetch_data(self.update_signal.emit)
            # TODO: put other heavy lifting works here
            # TODO: consider merge other worker threads to this one?
        except ValueError as ex:
            # from _fetch_data(), print it and quit
            print(ex, file=sys.stderr)
            print("[ERROR] possible reason: no image available for the selected detector/scan", file=sys.stderr)
        except Exception as ex:
            # use MainWindow's exception handler
            if self.exception_handler is not None:
                self.exception_handler(ex)

    def kill(self):
        pass

    def _save_h5(self, update_fcn=None):
        '''
        args = [db, param, scan_num, roi_width, roi_height, cx, cy, threshold, bad_pixels]
        '''
        print("saving data to h5, this may take a while...")
        save_data(*self.args)
        print("h5 saved.")

    def _fetch_data(self, update_fcn=None):
        '''
        args = [db, scan_id, det_name]
        '''
        if update_fcn is not None:
            print("loading begins, this may take a while...", end='')
            metadata = load_metadata(*self.args)

            # sanity checks
            if metadata['nz'] == 0:
                raise ValueError("nz = 0")
            #print("databroker connected, parsing experimental parameters...", end='')

            update_fcn(0, metadata) # 0 is just a placeholder


class PtychoReconFakeWorker(QtCore.QThread):
    update_signal = QtCore.pyqtSignal(int, object)

    def __init__(self, param:Param=None, parent=None):
        super().__init__(parent)
        self.param = param

    def _get_random_message(self, it):
        object_chi = np.random.random()
        probe_chi = np.random.random()
        diff_chi = np.random.random()
        return '[INFO] DM {:d} object_chi = {:f} probe_chi = {:f} diff_chi = {:f}'.format(
            it, object_chi, probe_chi, diff_chi)

    def _array_to_str(self, arr):
        arrstr = ''
        for v in arr: arrstr += '{:f} '.format(v)
        return arrstr

    def _get_random_message_multi(self, it):
        object_chi = np.random.random(4)
        probe_chi = np.random.random(4)
        diff_chi = np.random.random(4)

        object_chi_str = self._array_to_str(object_chi)
        probe_chi_str = self._array_to_str(probe_chi)
        diff_chi_str = self._array_to_str(diff_chi)

        return '[INFO] DM {:d} object_chi = {:s} probe_chi = {:s} diff_chi = {:s}'.format(
            it, object_chi_str, probe_chi_str, diff_chi_str)


    def _parse_message(self, message):
        message = str(message).replace('[', '').replace(']', '')

        tokens = message.split()
        id, alg, it = tokens[0], tokens[1], int(tokens[2])

        metric_tokens = tokens[3:]
        metric = {}
        name = 'Unknown'
        data = []

        for i in range(len(metric_tokens)):
            token = str(metric_tokens[i])

            if token == '=': continue

            if i < len(metric_tokens) - 2 and metric_tokens[i+1] == '=':
                if len(data): metric[name] = list(data)
                name = token
                data = []
                continue

            data.append(float(token))

        if len(data):
            metric[name] = data

        return id, alg, it, metric

    def run(self):
        from time import sleep
        update_fcn = self.update_signal.emit
        for it in range(self.param.n_iterations):

            message = self._get_random_message(it)
            _id, _alg, _it, _metric = self._parse_message(message)

            update_fcn(_it+1, _metric)
            sleep(.1)

        print("finished")

    def kill(self):
        pass
