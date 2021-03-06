cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_COMPILER "mpicxx")
set(CMAKE_C_COMPILER "mpicc")

project(_matpatch)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
set(CMAKE_CXX_FLAGS "-O3 -march=native -DLINUX -DNDEBUG")

set(CMAKE_EXE_LINKER_FLAGS "-mtune=native")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} 
    matpatch/matpatch.cpp
    matpatch/matinvert.cpp
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON)


find_package(PETSc REQUIRED)
message(STATUS ${PETSc})
find_package(PETSc4py)
message(STATUS ${PETSc4py})

target_link_libraries(${PROJECT_NAME} PRIVATE ${PETSC_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PETSC_INCLUDES} ${PETSC4PY_INCLUDE_DIRS})
