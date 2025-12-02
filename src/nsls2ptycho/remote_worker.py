import os,sys,socket,time,signal,subprocess,getpass
from .core.utils import *
from .core.ptycho.utils import *
from fcntl import fcntl, F_GETFL, F_SETFL
from os import O_NONBLOCK
import traceback
import numpy as np
from textwrap import dedent

# for frontend-backend communication
from posix_ipc import SharedMemory, ExistentialError
import mmap

SLURM_SERVER_NAME = 'orion'

class recon_worker:
    def exit(self,sig,frame):
        print('Ctrl+C!')
        self.msg_export('[Working]Aborting...')
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.abort_recon()
        sys.exit(0)
    
    def __init__(self,monitor_path,timeout):
        self.monitor_path = monitor_path
        self.timeout = timeout
        self.uuid = ''
        self.msg_file = os.path.join(os.path.join(self.monitor_path,'msg'+self.uuid))
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

        if not p.multislice_flag:
            self._prb = np.ndarray(shape=(p.n_iterations, p.prb_mode_num, p.nx, p.ny), dtype=datatype, buffer=self.mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.obj_mode_num, nx_obj, ny_obj), dtype=datatype, buffer=self.mm_list[2], order='C')
        else:
            self._prb = np.ndarray(shape=(p.n_iterations, 1, p.nx, p.ny), dtype=datatype, buffer=self.mm_list[1], order='C')
            self._obj = np.ndarray(shape=(p.n_iterations, p.slice_num, nx_obj, ny_obj), dtype=datatype, buffer=self.mm_list[2], order='C')
    
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
        except:
            # either not using GUI, monitor is turned off, global variables are deleted or not yet created!
            # need to examine the last case
            try:
                SharedMemory("/"+self.p.shm_name+"_obj_size").unlink()
                SharedMemory("/"+self.p.shm_name+"_prb").unlink()
                SharedMemory("/"+self.p.shm_name+"_obj").unlink()
            except:
                pass # nothing to clean up, we're done

    def msg_export(self,msg):
        print(msg)
        if os.path.isdir(self.monitor_path):
            if self.uuid != '':
                if not os.path.isfile(self.msg_file):
                    with open(self.msg_file,'w') as f:
                        pass
                with open(self.msg_file,'a') as f:
                    f.write(msg+'\n')
    def cleanup(self):
        self.close_mmap()
        if self.fname_full and os.path.exists(self.fname_full):
            os.remove(self.fname_full)
            self.fname_full = None
        if os.path.exists(os.path.join(self.monitor_path,f'prb_live{self.uuid}.npy')):
            os.remove(os.path.join(self.monitor_path,f'prb_live{self.uuid}.npy'))
        if os.path.exists(os.path.join(self.monitor_path,f'obj_live{self.uuid}.npy')):
            os.remove(os.path.join(self.monitor_path,f'obj_live{self.uuid}.npy'))

        
    def abort_recon(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
        self.msg_export('[Worker]Recon aborted')
        self.cleanup()

    def complete_recon(self):
        if self.fname_full:
            self.msg_export('[Worker]Recon done for '+self.fname)
            self.cleanup()
            # Remove msg file
            if os.path.isfile(os.path.join(self.monitor_path,'msg'+self.p.uuid)):
                os.remove(os.path.join(self.monitor_path,'msg'+self.p.uuid))

    def recon(self):
        with open(os.path.join(self.monitor_path,self.fname,),'w') as f:
            f.write('#running\n'+self.fcontent)
        self.fname_full = os.path.join(self.monitor_path,self.fname)
        self.p = parse_config(self.fname_full)

        self.uuid = self.p.uuid
        self.msg_file = os.path.join(os.path.join(self.monitor_path,'msg'+self.uuid))
        self.msg_export('[Worker]Start reconstructing '+self.fname)

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
                    if os.path.isfile(os.path.join(self.monitor_path,'abort'+self.uuid)):
                        self.process.terminate()
                        os.remove(os.path.join(self.monitor_path,'abort'+self.uuid))
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
                                np.save(os.path.join(self.monitor_path,f'prb_live{self.uuid}.npy'),self._prb[it-1])
                                np.save(os.path.join(self.monitor_path,f'obj_live{self.uuid}.npy'),self._obj[it-1])
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
        start_time = time.time()
        print('Ptycho worker started monitoring path '+self.monitor_path)
        self.dot_count = 1
        self.job_done = False
        while not self.job_done:
            if not os.path.isdir(self.monitor_path):
                print(f'\rWaiting for monitored path {self.monitor_path} to be created{"." * self.dot_count}   ',end='')
                self.dot_count = (self.dot_count)%3 + 1
                time.sleep(0.5)
            else:
                flist = [f for f in os.listdir(self.monitor_path) if f.startswith('ptycho')]
                if not flist:
                    print(f'\r[Worker]Recon folder is empty, waiting for task{"." * self.dot_count}   ',end='')
                    self.dot_count = (self.dot_count)%3 + 1
                    time.sleep(0.5)
                for fname in flist:
                    if os.path.isfile(os.path.join(self.monitor_path,fname)):
                        with open(os.path.join(self.monitor_path,fname,),'r') as f:
                            self.fcontent = f.read()
                        if not self.fcontent.startswith('#running'):
                            print('Loading jobfile '+fname)
                            self.fname = fname
                            self.recon()
                            if self.timeout is not None:
                                self.job_done = True
                                break
                        else:
                            time.sleep(0.5)
                            pass
            if self.timeout is not None and time.time() - start_time > self.timeout:
                self.job_done = True
                
class recon_worker_slurm:
    def __init__(self,slurm_header = None):
        self.base_dir = os.path.expanduser("~") + "/.ptycho_gui/"
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        if slurm_header is None:
            self.slurm_header = self.base_dir + "/.ptycho_slurm_job%s"
        else:
            self.slurm_header = slurm_header
        self.sbatch_header = self.base_dir + f"/ptycho_slurm%s.sh"
        self.slurm_exit_signal = self.base_dir + "/.ptycho_slurm_exit"

        # Exit previously running monitor threads
        try:
            print('Sending exit signal to existing slurm monitor threads...')
            os.listdir(self.base_dir)
            with open(self.slurm_exit_signal,'w') as f:
                f.write('EXIT')
            time.sleep(2)
            os.remove(self.slurm_exit_signal)
        except:
            pass

        self.dot_count = 1
        self.job_list = []
    
    def clear_slurm_headers(self,uuid):
        if os.path.isfile(self.slurm_header%uuid):
            try:
                os.remove(self.slurm_header%uuid)
            except:
                pass
        if os.path.isfile(self.sbatch_header%uuid):
            try:
                os.remove(self.sbatch_header%uuid)
            except:
                pass

        
    def exit(self,sig = None ,frame = None):
        print('\nExit signal received.')
        while len(self.job_list) > 0:
            self.query_jobs()
            for i in range(len(self.job_list)-1,-1,-1): # Iterate in reverse order to pop correctly
                job = self.job_list[i]
                self.clear_slurm_headers(job['uuid'])
                if job['status'] == '' or job['status'] == 'CG':
                    self.job_list.pop(i)
                elif job['status'] == 'PD':
                    # Cancel job and allocation
                    print(subprocess.run(['scancel',job['jobid']],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    ).stdout.decode('utf-8').strip())
                elif job['status'] == 'R':
                    # Send abort to worker
                    with open(os.path.join(self.remote_config_path,'abort'+job['uuid']),'w') as f:
                        pass
            time.sleep(0.5)
        sys.exit(0)
    
    def new_sbatch_job(self,fname):
        try:
            with open(fname,'r') as f:
                l = f.readlines()[0].split()
        except:
            return None

        if l is not None:
            uuid = fname[-4:]
            self.remote_config_path = l[0]
            nthreads = l[1]
            parent_module = '.'.join(__loader__.name.rsplit('.', 2)[:-1]) # get parent module name to run the correct recon worker
            sbatch_script_path = self.sbatch_header%uuid
            sbatch_script = dedent(f'''
                    #!/bin/bash
                    #SBATCH --job-name=ptycho
                    #SBATCH --qos=normal
                    #SBATCH --time=0-03:00:00

                    #SBATCH --ntasks-per-node={nthreads}
                    #SBATCH --gres=gpu:{nthreads}

                    #SBATCH --partition=normal
                    #SBATCH --error={self.base_dir + "/.ptycho_slurm.err"}
                    #SBATCH --output={self.base_dir + "/.ptycho_slurm.out"}
                    source load-hxn
                    python -W ignore -m {parent_module}.remote_worker {self.remote_config_path} 5 # <monitor_path> <timeout>
                ''').strip()
            
            with open(sbatch_script_path,'w') as f:
                f.write(sbatch_script)

            sbatch_command = ["sbatch","--parsable",sbatch_script_path]
            
            print("")
            print(sbatch_command)

            jobid = subprocess.run(
                sbatch_command,
                stdout=subprocess.PIPE
            ).stdout.decode('utf-8').strip()

            self.job_list.append({'uuid':uuid,'jobid':jobid,'status':'PD'})

            return jobid
        else:
            return None

    def query_jobs(self):
        for job in self.job_list:
            squeue_query_command = f"squeue -j {job['jobid']} -h --format=%t".split()
            job['status'] = subprocess.run(
                        squeue_query_command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    ).stdout.decode('utf-8').strip()

    def monitor(self):
        last_update = -10000
        while True:
            flist = [f for f in os.listdir(self.base_dir) if f.startswith('.ptycho_slurm_job')]
            for fname in flist:
                uuid = fname[-4:]
                if not any(job['uuid'] == uuid for job in self.job_list):
                    jid = self.new_sbatch_job(os.path.join(self.base_dir,fname))

                    if jid is not None:
                        print(f"Submitted batch job {self.job_list[-1]['jobid']}")
                        time.sleep(1)
            
            self.query_jobs()

            if len(self.job_list) == 0:
                print(f'\rWaiting for slurm task{"." * self.dot_count}   ',end='')
                self.dot_count = (self.dot_count)%3 + 1
            else:
                n_running = 0
                n_pending = 0

                for i in range(len(self.job_list)-1,-1,-1): # Iterate in reverse order to pop correctly
                    job = self.job_list[i]
                    if job['status'] == '' or job['status'] == 'CG': # Complete
                        self.clear_slurm_headers(job['uuid'])
                        self.job_list.pop(i)
                    elif job['status'] == 'PD': # Pending allocation
                        n_pending += 1
                    elif job['status'] == 'R': # Running
                        n_running += 1

                if n_pending + n_running > 0:
                    print('\rYou have',end='')
                    if n_running > 0 :
                        print(f' {n_running} job running',end='')
                    if n_pending > 0 :
                        print(f' {n_running} job pending allocation',end='')
                    
                    print(f'{"." * self.dot_count}   ',end='')
                    self.dot_count = (self.dot_count)%3 + 1
                    
            # Show the queue every 30 s
            if (time.time() - last_update)>30:
                print(subprocess.run('squeue',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                ).stdout.decode('utf-8'))
                print('# squeue display updates every 30s #')
                last_update = time.time()
            time.sleep(0.5)
                        
            os.listdir(self.base_dir)
            if os.path.isfile(self.slurm_exit_signal):
                os.remove(self.slurm_exit_signal)
                self.exit()
                    


            
def main():
    srv_name = socket.gethostname().split('.')[0]

    if srv_name.startswith(SLURM_SERVER_NAME):
        print(f'{srv_name} is a slurm allocation server, running in slurm monitor mode...')
        r = recon_worker_slurm()
        signal.signal(signal.SIGINT,r.exit)
        r.monitor()
    else:
        if len(sys.argv) == 1: # started without argument
            # monitor current folder
            monitor_path = os.path.join(os.path.abspath('.'),'remote_'+srv_name+'_'+getpass.getuser())
            timeout = None
        else: # First argument is monitor_path
            monitor_path = sys.argv[1]
            if len(sys.argv) == 3: # Second argument is timeout time in second
                timeout = int(sys.argv[2])
            else:
                timeout = None
        r = recon_worker(monitor_path,timeout)
        signal.signal(signal.SIGINT,r.exit)
        r.monitor()

if __name__ == '__main__':
    main()
