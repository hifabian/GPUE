cmake_minimum_required(VERSION 3.12)
project(gpue_python_module LANGUAGES CXX CUDA)

if(${ENABLE_PYTHON})
    set(CMAKE_CXX_STANDARD 14)
    set(PYBIND11_CPP_STANDARD -std=c++14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(Python COMPONENTS Interpreter Development)

    # Copied from https://stackoverflow.com/questions/51907755/building-a-pybind11-module-with-cpp-and-cuda-sources-using-cmake
    #add_library(_PyGPUE SHARED gpue_pybind11.cu)
    #target_link_libraries(_PyGPUE PRIVATE pybind11::module gpue CUDA::cufft CUDA::cudart ${HDF5_CXX_LIBRARIES})
    #set_target_properties(_PyGPUE PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
    #set_target_properties(_PyGPUE PROPERTIES COMPILE_FLAGS "-dc")
    #set_target_properties(_PyGPUE PROPERTIES 
    #    POSITION_INDEPENDENT_CODE ON
    #    CUDA_VISIBILITY_PRESET "hidden"
    #    CUDA_SEPARABLE_COMPILATION ON
    #)

    add_subdirectory(io)
    add_subdirectory(simulator)
    add_subdirectory(vortex)

    #Ensure the Python module init and install files are copied to build directory
    configure_file(__init__.py ${CMAKE_CURRENT_BINARY_DIR}/__init__.py)
endif()

