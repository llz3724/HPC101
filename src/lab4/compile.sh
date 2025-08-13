#!/bin/bash

# rm -rf build
# mkdir build
cd build

# source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi
spack load intel-oneapi-compilers
spack load intel-oneapi-itac

cmake .. \
    -D CMAKE_C_COMPILER=mpiicx \
    -D CMAKE_CXX_COMPILER=mpiicpx \
    -D CMAKE_Fortran_COMPILER=mpiifx \
    -D CMAKE_EXE_LINKER_FLAGS="-trace"

make
