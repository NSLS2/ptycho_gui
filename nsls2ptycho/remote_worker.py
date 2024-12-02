import os,sys,socket,time,signal,subprocess
from .core.utils import *
from .core.ptycho.utils import *
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
import traceback
import numpy as np

# for frontend-backend communication
from posix_ipc import SharedMemory, ExistentialError
import mmap

class recon_worker:
    def exit(self,sig,frame):
        print('Ctrl+C!')
        self.msg_export('[Working]Aborting...')
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.abort_recon()
        sys.exit(0)
    
    def __init__(self,work_path):
        self.work_path = work_path
        self.srv_name = socket.gethostname().split('.')[0]
        self.monitor_path = os.path.join(os.path.abspath(self.work_path),'remote_'+self.srv_name)
        self.msg_file = os.path.join(os.path.join(self.monitor_path,'msg'))
        self.fname = None
        self.fname_full = None
        self.process = None

    def init_mmap(self):
        p = self.p
        datasize = 8 if p.precision == 'single' else 16
        datatype = np.complex64 if p.precision == 'single' else np.complex128

        self.mm_list = []
        self.shm_list = []
        for i, name in enumerate(["/"+p.shm_name+"_obj_size", "/"+p.shm_name+"_prb", "/"+p.shm_name+"_obj"]):
            #print(name)
            self.shm_list.append(SharedMemory(name))
            self.mm_list.append(mmap.mmap(self.shm_list[i].fd, self.shm_list[i].size))

        nx_obj = int.from_bytes(self.mm_list[0].read(8), byteorder='big')
        ny_obj = int.from_bytes(self.mm_list[0].read(8), byteorder='big') # the file position has been moved by 8 bytes when we get nx_obj

        if p.mode_flag:
            self._prb = np.ndarray(shape=(p.n_iterations, p.prb_mode_num, p.nx, p.ny), dtype=datatype, buffer=self.mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.obj_mode_num, nx_obj, ny_obj), dtype=datatype, buffer=self.mm_list[2], order='C')
        elif p.multislice_flag:
            self._prb = np.ndarray(shape=(p.n_iterations, 1, p.nx, p.ny), dtype=datatype, buffer=self.mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.slice_num, nx_obj, ny_obj), dtype=datatype, buffer=self.mm_list[2], order='C')
        else:
            self._prb = np.ndarray(shape=(p.n_iterations, 1, p.nx, p.ny), dtype=datatype, buffer=self.mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, 1, nx_obj, ny_obj), dtype=datatype, buffer=self.mm_list[2], order='C')
    
    def close_mmap(self):
        # We close shared memory as long as the backend is terminated either normally or 
        # abnormally. The subtlety here is that the monitor should still be able to access
        # the intermediate results after mmaps' are closed. A potential segfault is avoided 
        # by accessing the transformed results, which are buffered, not the original ones.
        try:
            for mm, shm in zip(self.mm_list, self.shm_list):
                mm.close()
                shm.close_fd()
                shm.unlink()
            self.mm_list = []
            self.shm_list = []
        except NameError:
            # either not using GUI, monitor is turned off, global variables are deleted or not yet created!
            # need to examine the last case
            try:
                SharedMemory("/"+self.p.shm_name+"_obj_size").unlink()
                SharedMemory("/"+self.p.shm_name+"_prb").unlink()
                SharedMemory("/"+self.p.shm_name+"_obj").unlink()
            except ExistentialError:
                pass # nothing to clean up, we're done

    def msg_export(self,msg):
        print(msg)
        if os.path.isdir(self.monitor_path):
            if not os.path.isfile(self.msg_file):
                with open(self.msg_file,'w') as f:
                    pass
            with open(self.msg_file,'a') as f:
                f.write(msg+'\n')

    def abort_recon(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.msg_export('[Worker]Recon aborted')
        if self.fname_full:
            os.remove(self.fname_full)
            self.fname_full = None

    def complete_recon(self):
        if self.fname_full:
            self.msg_export('[Worker]Recon done for '+self.fname)
            os.remove(self.fname_full)
            # Clear msg file
            with open(self.msg_file,'w') as f:
                pass
            self.fname_full = None

    def recon(self):
        self.msg_export('[Worker]Start reconstructing '+self.fname)
        with open(os.path.join(self.monitor_path,self.fname,),'w') as f:
            f.write('#running\n'+self.fcontent)
        self.fname_full = os.path.join(self.monitor_path,self.fname)
        self.p = parse_config(self.fname_full)

        nthreads = len(self.p.gpus) if self.p.gpu_flag else 1

        parent_module = '.'.join(__loader__.name.rsplit('.', 2)[:-1]) # get parent module name to run the correct recon worker
        mpirun_command = ["mpirun", "-n", str(nthreads), "python", "-W", "ignore", "-m",parent_module+".core.ptycho.recon_ptycho_gui",self.fname_full]

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
                flags = fcntl(run_ptycho.stderr, F_GETFL) # first get current stderr flags
                fcntl(run_ptycho.stderr, F_SETFL, flags | O_NONBLOCK)

                while True:
                    if os.path.isfile(os.path.join(self.monitor_path,'abort')):
                        self.process.terminate()
                        os.remove(os.path.join(self.monitor_path,'abort'))
                        raise Exception("Server sends abort signal")
                    stdout = run_ptycho.stdout.readline()
                    stderr = run_ptycho.stderr.readline() # without O_NONBLOCK this will very likely block
                    
                    if (run_ptycho.poll() is not None) and (stdout==b'') and (stderr==b''):
                        break

                    if stdout:
                        stdout = stdout.decode('utf-8')
                        self.msg_export(stdout.strip())
                        tokens = stdout.split()
                        if len(tokens) > 2 and tokens[0] == "[INFO]":
                            it = int(tokens[2])
                            if (it-1) % self.p.display_interval == 0:
                                np.save(os.path.join(self.monitor_path,'prb_live.npy'),self._prb)
                                np.save(os.path.join(self.monitor_path,'obj_live.npy'),self._obj)
                        if len(tokens) == 3 and tokens[0] == "shared":
                            self.init_mmap()

                    if stderr:
                        stderr = stderr.decode('utf-8')
                        self.msg_export(stderr.strip())

                # get the return value 
                self.return_value = run_ptycho.poll()

            if self.return_value != 0:
                message = "At least one MPI process returned a nonzero value, so the whole job is aborted.\n"
                message += "If you did not manually terminate it, consult the Traceback above to identify the problem."
                raise Exception(message)
        except Exception as ex:
            self.msg_export(str(ex).strip())
            traceback.print_exc()
            self.abort_recon()
            #print(ex, file=sys.stderr)
            #raise ex
        finally:
            # clean up temp file
            filepath = self.p.working_directory + "/." + self.p.shm_name + ".txt"
            if os.path.isfile(filepath):
                os.remove(filepath)
            self.complete_recon()
            


    def monitor(self):
        print('Ptycho worker started monitoring path '+self.monitor_path)
        while True:
            if not os.path.isdir(self.monitor_path):
                print('Waiting for monitored path to be created...')
            else:
                flist = [f for f in os.listdir(self.monitor_path) if f.startswith('ptycho')]
                if not flist:
                    self.msg_export("[Worker]Recon folder is empty, waiting for task...")
                for fname in flist:
                    print('Loading jobfile '+fname)
                    with open(os.path.join(self.monitor_path,fname,),'r') as f:
                        self.fcontent = f.read()
                    if not self.fcontent.startswith('#running'):
                        self.fname = fname
                        self.recon()
                    else:
                        print(__loader__.name)
                        self.msg_export('[Warning]Another session of ptycho worker is running on this server or the previous worker didn\'t exit normally')
            time.sleep(3)

if __name__ == '__main__':
    r = recon_worker('.')
    signal.signal(signal.SIGINT,r.exit)
    r.monitor()