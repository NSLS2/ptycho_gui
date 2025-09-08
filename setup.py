import os
import re
import sys
import traceback
from setuptools import setup

## CPU codes are currently disabled as they haven't been maintained since version 2.0.0
# cython is needed for compiling the CPU codes
# try:
#     from Cython.Build import cythonize
# except ImportError:
#     print("\n************************************************************************\n"
#           "***** Cython is not found. Use the corresponding C source instead. *****\n" 
#           "************************************************************************\n", file=sys.stderr)
#     from distutils.extension import Extension
#     from glob import glob
#     extensions = []
#     # this doesn't work because setuptools doesn't support glob pattern...
#     #extensions = [Extension("*", ["nsls2ptycho/core/ptycho/*.c"])]
#     for filename in glob("nsls2ptycho/core/ptycho/*.c"):
#         mod = os.path.basename(filename)[:-2]
#         extensions.append(Extension("nsls2ptycho.core.ptycho."+mod, [filename]))
# else:
#     extensions = cythonize("nsls2ptycho/core/ptycho/*.pyx")

REQUIREMENTS = ['mpi4py', 'pyfftw', 'numpy', 'nvtx', 'scipy', 'matplotlib', 'Pillow', 'h5py', 'posix_ipc', 'h5py>=3.9.0']

# see if PyQt5 is already installed --- pip and conda use different names...
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    REQUIREMENTS.append('PyQt5')

# Check if cupy exists
try:
    import cupy
except ImportError:
    print("CuPy not found. Will install...", file=sys.stderr)
    try:
        with os.popen('nvidia-smi') as stream:
            nv_version = stream.read()
        match = re.search(r'CUDA Version+:\s+(\d+\.+\d)',nv_version)
        cuda_version = match.group(1)
        print(f'Cuda version {cuda_version} detected')
        cupy_package = 'cupy-cuda'+cuda_version.split('.')[0]+'x'
        print(f'{cupy_package} will be installed')
        REQUIREMENTS.append(cupy_package)
    except:
        print("\n************************************************************************\n"
              "**** Unable to detect cuda version, please install cupy-cuda{version}x package manually to run GPU reconstruction. ****\n"
              "************************************************************************\n", file=sys.stderr)

# ...and then if numba exists
try:
    import numba
except ImportError:
    REQUIREMENTS.append('numba>=0.41.0') # for bug fix in __cuda_array_interface__

# Get __version__ variable
exec(open(os.path.join(os.path.dirname(__file__),'src', 'nsls2ptycho', '_version.py')).read())

setup(version=__version__,
      install_requires=REQUIREMENTS
      )
