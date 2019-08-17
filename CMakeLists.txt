cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_C_COMPILER "mpicc")

project(_matpatch)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
set(CMAKE_CXX_FLAGS "-O3 -march=native -DLINUX -DNDEBUG")

set(CMAKE_EXE_LINKER_FLAGS "-mtune=generic")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} 
    matpatch/matpatch.cpp
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)


find_library(MKL mkl_sequential PATHS /opt/intel/mkl/lib/ REQUIRED)
message(STATUS ${MKL})

find_package(LAPACK REQUIRED)
message(STATUS ${LAPACK_LIBRARIES})

find_package(PETSc REQUIRED)
message(STATUS ${PETSc})

target_link_libraries(${PROJECT_NAME} PRIVATE ${MKL} ${LAPACK_LIBRARIES} ${PETSC_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PETSC_INCLUDES})
