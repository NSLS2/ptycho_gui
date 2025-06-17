# ---------------- package metadata ----------------
NAME = 'nsls2ptycho'
DESCRIPTION = 'NSLS-II Ptychography Software'
AUTHOR = 'Leo Fang, Zirui Gao, Sungsoo Ha, Zhihua Dong, and Xiaojing Huang'
EMAIL = 'xjhuang@bnl.gov'
LINK = 'https://github.com/NSLS2/ptycho_gui'
LICENSE = 'MIT'
REQUIREMENTS = ['mpi4py', 'pyfftw', 'numpy', 'scipy', 'matplotlib', 'Pillow', 'h5py', 'posix_ipc']
# --------------------------------------------------

import os
import sys
from setuptools import setup #, find_packages
import numpy

# cython is needed for compiling the CPU codes
try:
    from Cython.Build import cythonize
except ImportError:
    print("\n************************************************************************\n"
          "***** Cython is not found. Use the corresponding C source instead. *****\n"
          "************************************************************************\n", file=sys.stderr)
    from distutils.extension import Extension
    from glob import glob
    extensions = []
    # this doesn't work because setuptools doesn't support glob pattern...
    #extensions = [Extension("*", ["nsls2ptycho/core/ptycho/*.c"])]
    for filename in glob("nsls2ptycho/core/ptycho/*.c"):
        mod = os.path.basename(filename)[:-2]
        extensions.append(Extension("nsls2ptycho.core.ptycho."+mod, [filename]))
else:
    extensions = cythonize("nsls2ptycho/core/ptycho/*.pyx")

# see if PyQt5 is already installed --- pip and conda use different names...
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    REQUIREMENTS.append('PyQt5')

# for generating .cubin files
# TODO: add a flag to do this only if GPU support is needed?
import nsls2ptycho.core.ptycho.build_cuda_source as bcs
cubin_path = bcs.compile()

# if GPU support is needed, check if cupy exists
if len(cubin_path) > 0:
    # skip depending CuPy on OS X as the wheel is not provided
    if not bcs.PLATFORM_DARWIN:
        cuda_ver = str(bcs._cuda_version)
        major = int(cuda_ver[:-2])//10
        minor = int(cuda_ver[-2:])//10
        if major > 10:
            # Newer versions of CuPy are published as cupy-cuda11x, cupy-cuda12x, etc
            minor = "x"
            # Older versions of CuPy are published as cupy-cuda80, cupy-cuda102, etc
        major = str(major)
        minor = str(minor)
        try:
            import cupy
        except ImportError:
            cupy_ver = 'cupy-cuda'+major+minor
            print("CuPy not found. Will install", cupy_ver+"...", file=sys.stderr)
            REQUIREMENTS.append(cupy_ver+'>=6.0.0') # for experimental FFT plan feature and bug fix in __cuda_array_interface__
    # ...and then if numba exists
    try:
        import numba
    except ImportError:
        REQUIREMENTS.append('numba>=0.41.0') # for bug fix in __cuda_array_interface__

# Get __version__ variable
exec(open(os.path.join(os.path.dirname(__file__), 'nsls2ptycho', '_version.py')).read())

# start building
with open("README.md", "r") as f:
    long_description = f.read()

setup(name=NAME,
      version=__version__,
      #packages=find_packages(),
      packages=["nsls2ptycho", "nsls2ptycho.core", "nsls2ptycho.ui", "nsls2ptycho.core.ptycho", "nsls2ptycho.core.widgets"],
      entry_points={
          'gui_scripts': ['run-ptycho = nsls2ptycho.ptycho_gui:main'],
          'console_scripts': ['run-ptycho-backend = nsls2ptycho.core.ptycho.recon_ptycho_gui:main']
      },
      install_requires=REQUIREMENTS,
      #extras_require={'GPU': 'cupy'}, # this will build cupy from source, may not be the best practice!
      ext_modules=extensions,
      include_dirs=[numpy.get_include()],
      #dependency_links=['git+https://github.com/leofang/ptycho.git#optimization']
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type="text/markdown",
      author=AUTHOR,
      author_email=EMAIL,
      url=LINK,
      license=LICENSE,
      include_package_data=True, # to include all precompiled .cubin files
      )
