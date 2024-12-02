import getpass
import os
from pwd import getpwuid
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI

from .ptycho.utils import split


# DEPRECARED: migrated to nsls2ptycho.core.widgets.mplcanvas
def plot_point_process_distribution(pts, mpi_size, colormap=cm.jet):
    '''
    Plot N scanning points in mpi_size different colors

    Parameters:
        - pts: np.array([[x0, x1, ..., xN], [y0, y1, ..., yN]])
        - mpi_size: number of MPI processes
        - colormap 
    '''
    a = split(pts.shape[1], mpi_size)
    colors = colormap(np.linspace(0, 1, len(a)))
    for i in range(mpi_size):
        plt.scatter(pts[0, a[i][0]:a[i][1]], pts[1, a[i][0]:a[i][1]], c=colors[i])
    plt.show()


def find_owner(filename):
    # from https://stackoverflow.com/a/1830635
    return getpwuid(os.stat(filename).st_uid).pw_name


def clean_shared_memory(pid=None):
    '''
    This function cleans up shared memory segments created by the GUI
    or a buggy Open MPI or CUDA.
    '''
    # this only works for linux that has /dev/shm
    if not sys.platform.startswith('linux'):
        print("This function works only under Linux. Stop.", file=sys.stderr)
        return
    assert os.path.isdir('/dev/shm/')

    from posix_ipc import SharedMemory
    shm_list = os.listdir('/dev/shm/')
    user = getpass.getuser()   

    for shm in shm_list:
        if (shm.startswith('ptycho') \
            or shm.startswith('vader') \
            or shm.startswith('cuda')) \
           and user == find_owner('/dev/shm/'+shm):

            if (pid is None) or (pid is not None and pid in shm):
                s = SharedMemory("/"+shm)
                s.close_fd()
                s.unlink()

    print("Done.")


def get_mpi_num_processes(mpi_file_path):
    # use MPI machine file if available, assuming each line of which is: 
    # ip_address slots=n max-slots=n   --- Open MPI
    # ip_address:n                     --- MPICH, MVAPICH
    with open(mpi_file_path, 'r') as f:
        node_count = 0
        if MPI.get_vendor()[0] == 'Open MPI':
            for line in f:
                line = line.split()
                node_count += int(line[1].split('=')[-1])
        elif 'MPICH' in MPI.get_vendor()[0] or 'MVAPICH' in MPI.get_vendor()[0]:
            for line in f:
                line = line.split(":")
                node_count += int(line[1])
        else:
            raise RuntimeError("mpi4py is built on top of unrecognized MPI library. "
                               "Only Open MPI, MPICH, and MVAPICH are tested.")

    return node_count


def use_mpi_machinefile(mpirun_command, mpi_file_path):
    # use MPI machine file if available, assuming each line of which is: 
    # ip_address slots=n max-slots=n   --- Open MPI
    # ip_address:n                     --- MPICH, MVAPICH
    node_count = get_mpi_num_processes(mpi_file_path)

    if MPI.get_vendor()[0] == 'Open MPI':
        mpirun_command.insert(3, "-machinefile")
        # use mpirun to find where MPI is installed
        import shutil
        path = os.path.split(shutil.which('mpirun'))[0] 
        if path.endswith('bin'):
            path = path[:-3]
        mpirun_command[4:4] = ["--prefix", path, "-x", "PATH", "-x", "LD_LIBRARY_PATH"]
    elif 'MPICH' in MPI.get_vendor()[0] or 'MVAPICH' in MPI.get_vendor()[0]:
        mpirun_command.insert(3, "-f")
    else:
        raise RuntimeError("mpi4py is built on top of unrecognized MPI library. "
                           "Only Open MPI, MPICH, and MVAPICH are tested.")
    mpirun_command[2] = str(node_count) # use all available nodes
    mpirun_command.insert(4, mpi_file_path)

    return mpirun_command


def set_flush_early(mpirun_command):
    if 'MPICH' in MPI.get_vendor()[0] or 'MVAPICH' in MPI.get_vendor()[0]:
        mpirun_command.insert(-2, "-u") # force flush asap (MPICH is weird...)
    return mpirun_command

def parse_range2(batch_items, batch_processing = True):
    '''
    Note the range is inclusive on both ends.
    Ex: 1238 - 1242 with step size 2 --> [1238, 1240, 1242]
    '''
    scan_range = []
    scan_numbers = []

    if batch_processing and not batch_items:
        raise ValueError("No item list is given for batch processing.")

    # first parse items and separate them into two catogories
    slist = batch_items.split(',')
    for item in slist:
        if ':' in item:
            sublist = np.arange(0,1e6,dtype=np.int32)[slice(*map(lambda x: int(x.strip()) if x.strip() else None,item.split(':')))]
            scan_numbers += list(sublist)
        elif len(item) == 0:  # for empty string
            continue
        else:
            scan_numbers.append(int(item.strip()))

    if batch_processing:
        scan_numbers.sort(reverse=True)

    return scan_numbers


def parse_range(batch_items, every_nth_item = 1, batch_processing = True):
    '''
    Note the range is inclusive on both ends.
    Ex: 1238 - 1242 with step size 2 --> [1238, 1240, 1242]
    '''
    scan_range = []
    scan_numbers = []

    if batch_processing and batch_items == '':
        raise ValueError("No item list is given for batch processing.")

    # first parse items and separate them into two catogories
    slist = batch_items.split(',')
    for item in slist:
        if '-' in item:
            sublist = item.split('-')
            scan_range.append((int(sublist[0].strip()), int(sublist[1].strip())))
        elif len(item) == 0:  # for empty string
            continue
        else:
            scan_numbers.append(int(item.strip()))

    # next generate all legit items from the chosen ranges and make a sorted item list
    for item in scan_range:
        scan_numbers = scan_numbers + list(range(item[0], item[1]+1, every_nth_item))

    if batch_processing:
        scan_numbers.sort(reverse=True)

    return scan_numbers
