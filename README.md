# BNL NSLS-II in-house ptychography software
## Introduction

## Installation
While one can `pip install` this pacakge directly, most likely the non-Python dependencies will not be available. For the time being, therefore, we recommend using Conda.

### Installation on NSLS-II Beamline Machines
The instruction below is **for admins who have the root priviledge** to install the software at the system level so that *all* users logging in the machine can run the software directly without any setup.

#### Fully Automatic Installation (recommended)
*Note: The instructions for automatic installation are out of date and may not work as expected.*
1. Create a new conda environment named `ptycho_production`: `sudo /opt/conda/bin/conda create -p /opt/conda_envs/ptycho_production python=3.6 nsls2ptycho` (If you need beamline-specific packages, such as `hxntools` for HXN, append the package names in the `conda create` command. This helps resolve possible conflict/downgrade issues.)
The conda environment `ptycho_production` is activated under the hood using the `run-ptycho` script to be installed in the last step.
2. `fix_conda_privileges.sh`
2. `sudo -i` (switch to `root`)
3. `/opt/conda_envs/ptycho_production/bin/pip install 'cupy-cudaXX>=6.0.0b3'`, where `XX` is your CUDA toolkit version, available from `nvcc --version`
3. If needed, copy the script `run-ptycho` in the root of this repo to `/usr/local/bin/`: `sudo cp ./run-ptycho /usr/local/bin/`

To update the software, simple do `sudo conda update -n ptycho_production nsls2ptycho`

#### Manual Installation
1. Create a new conda environment named `ptycho_production`: `sudo conda create -n ptycho_production -c conda-forge python=3.9 pyfftw pyqt=5 'numpy<2' scipy matplotlib pillow h5py 'databroker<=1.999' openmpi mpi4py cython`. If you need beamline-specific packages, such as `hxntools` for HXN, append the package names in the `conda create` command. This helps resolve possible conflict/downgrade issues.
The conda environment `ptycho_production` is activated under the hood using the `run-ptycho` script to be installed in Step 9.
2. Activate the environment: `conda activate ptycho_production`.
3. Install additional packages using pip: `pip install posix_ipc`.
4. Make sure you are granted access to the backend, currently hosted in [this private GitHub repo](https://github.com/NSLS-II/ptycho).
5. Create a temporary workspace: `mkdir /tmp/build_ptycho; cd /tmp/build_ptycho`
6. Clone from the mirror of this repo: `git clone --recursive https://github.com/NSLS-II/ptycho_gui.git` (During the process `git` may prompt you to enter your GitHub login and password for cloning the backend.)
7. Move this repo to `/usr/local/`: `sudo mv ./ptycho_gui /usr/local/; cd /usr/local/ptycho_gui; rmdir /tmp/build_ptycho`
8. Install the GUI in "develop" mode: `sudo /opt/conda_envs/ptycho_production/bin/pip install -e .`
9. Copy the script `run-ptycho` to `/usr/local/bin/`: `sudo cp ./run-ptycho /usr/local/bin/`

To update the software, simple go to the code location and do `git pull` there. Since we installed in the develop mode (with `-e` flag) the files are symlinked to the conda env, so any updates we do to the code will be immediately up online. This can also work as a way to do "hot fixes".
```shell
cd /usr/local/ptycho_gui/
sudo git pull origin master    # update frontend
cd ./nsls2ptycho/core/ptycho/
sudo git pull origin master    # update backend
```

### Installation on Personal Machines
The procedure is similar to **Manual Installation** outlined above, except that it does not require `sudo`:
1. Make sure you are granted access to the backend, currently hosted in [this private GitHub repo](https://github.com/NSLS-II/ptycho)
2. `git clone --recursive https://github.com/NSLS-II/ptycho_gui.git` (during the process `git` will prompt you to enther your GitHub id and password for cloning the backend)
3. Create the conda environment as described in steps 1-3 of instructions for **Manual Installation** (do not use `sudo`). Alternatively, the needed packages may be installed in the existing Conda environment.
4. Activate the Conda environment.
5. Enter the cloned directory: `cd ./ptycho_gui`.
5. `pip install .` (or `pip install -e .`).

## Execution
1. Start the GUI: `run-ptycho`
2. Spawn two MPI processes without GUI: `mpirun -n 2 run-ptycho-backend input_file`

## Conventions
1. The GUI writes a config file to `~/.ptycho_gui/`
2. Once the working directory is specified in the GUI, it assumes that all HDF5 files are stored there, and the outputs are written to `working_dir/recon_results/SXXXXX/`, where `XXXXX` is the scan-number string.
3. A few compiled `.cubin` files are stored with the Python code

## References
- *High-Performance Multi-Mode Ptychography Reconstruction on Distributed GPUs*, Z. Dong, Y.-L. L. Fang *et al.*, 2018 NYSDS, DOI:[10.1109/NYSDS.2018.8538964](https://doi.org/10.1109/NYSDS.2018.8538964)

For users using the new solvers mADMM, PM, and APG, you are advised to cite additionlly the following paper:

- *Ptychographic phase retrieval by proximal algorithms*, H. Yan, [New J. Phys. 22 023035 (2020)](https://doi.org/10.1088/1367-2630/ab704e).


## License
MIT (subject to change)

Users are encouraged to cite the references above.

## Maintainer
- Leo Fang ([@leofang](https://github.com/leofang))
