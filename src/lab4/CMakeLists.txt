cmake_minimum_required(VERSION 3.12)
project(bicgstab C CXX Fortran)

find_package(OpenMP REQUIRED)
# Uncomment below is MPI env is available
find_package(MPI REQUIRED)

# Set Compiler
set(CMAKE_C_COMPILER "mpiicx")
set(CMAKE_CXX_COMPILER "mpiicpx")
set(CMAKE_Fortran_COMPILER "mpiifx")

# Set flags
set(CMAKE_C_FLAGS "-Og -g -qopenmp")
set(CMAKE_CXX_FLAGS "-Og -g -qopenmp")
set(CMAKE_Fortran_FLAGS "-Og -g -qopenmp")

# Include headers
include_directories(include)

file(GLOB COMMON_SOURCES "src/*.cpp")

# If found C implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.c")
  # 将 C++ 源文件 (${COMMON_SOURCES}) 和 C 源文件一起添加到可执行文件
  # CMake 会因为看到 .cpp 文件而自动选择 C++ 编译器进行链接，从而解决问题
  add_executable(bicgstab ${COMMON_SOURCES} src/bicgstab/solver.c)
endif()

# If found Fortran implementation, add it to the build
if(EXISTS "${CMAKE_SOURCE_DIR}/src/bicgstab/solver.f90")
  add_executable(bicgstab-fortran src/bicgstab/solver.f90 $<TARGET_OBJECTS:judger>)
endif()
