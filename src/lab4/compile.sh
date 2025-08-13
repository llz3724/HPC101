#!/bin/bash

# 脚本开始时先清理旧的构建目录，确保完全重新配置
echo "--- Cleaning previous build directory ---"
rm -rf build
mkdir build
cd build

# 加载编译环境
echo "--- Loading Spack environment ---"
# source /pxe/opt/spack/share/spack/setup-env.sh
spack load intel-oneapi-mpi
spack load intel-oneapi-compilers

# 运行 cmake，并通过 -D 参数强制指定 MPI 编译器。
# 这是最可靠的方法，可以覆盖任何可能冲突的环境变量。
echo "--- Configuring CMake with specific MPI compilers ---"
cmake .. \
    -D CMAKE_C_COMPILER=mpiicx \
    -D CMAKE_CXX_COMPILER=mpiicpx \
    -D CMAKE_Fortran_COMPILER=mpiifx

# 检查 CMake 配置是否成功
if [ $? -ne 0 ]; then
    echo "CMake configuration failed. Please check the output above."
    exit 1
fi

# 编译项目
echo "--- Building the project with make ---"
make

if [ $? -ne 0 ]; then
    echo "Build failed. Please check the output above."
    exit 1
fi

echo "--- Build successful ---"