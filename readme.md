# MPITS
Multipixel InSAR time series

This python library implements the method described in [Jolivet & Simons 2018](http://onlinelibrary.wiley.com/doi/10.1002/2017GL076533/full) (*Geophysical Research Letters*). In short, this library allows to solve the InSAR time series problem considering all pixels simultaneously and accounting for long wavelength trends and complete covariances. 

## Installation

Sorry, there is no setup.py script...

1. Start by installing the prerequisites: 
  - openmpi (maybe mpich... should be okay... has not been tested...)
  - PETSc
  - HDF5 library with parallel support
  - Python (tested with 2.7, but should be compatible with 3.6) with the following libraries:
    - numpy
    - scipy
    - mpi4py
    - h5py (linked with HDF5 with parallel support)
    - petsc4py 
  
2. Clone mpits in a directory
3. Add the appropriate directory to your PYTHONPATH 

## Examples

For examples, please visit [R. Jolivet's website](http://www.geologie.ens.fr/~jolivet).

