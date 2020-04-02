################################################################################

cmake_minimum_required(VERSION 3.17)
set(GPUE_TEXT "
    ▄▄▄▄   ▄▄▄▄▄▄    ▄▄    ▄▄  ▄▄▄▄▄▄▄▄ 
  ██▀▀▀▀█  ██▀▀▀▀█▄  ██    ██  ██▀▀▀▀▀▀ 
 ██        ██    ██  ██    ██  ██       
 ██  ▄▄▄▄  ██████▀   ██    ██  ███████  
 ██  ▀▀██  ██        ██    ██  ██       
  ██▄▄▄██  ██        ▀██▄▄██▀  ██▄▄▄▄▄▄ 
    ▀▀▀▀   ▀▀          ▀▀▀▀    ▀▀▀▀▀▀▀▀ 
")
message(${GPUE_TEXT})
project(gpue LANGUAGES C CXX CUDA)

# Echo makefile compilation commands
set(CMAKE_VERBOSE_MAKEFILE ON)

# Require build directory; forbid modifying the source files during compilation
set(CMAKE_DISABLE_IN_SOURCE_BUILD OFF)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

# Ensure C++11 enabled
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use release build type (performance optimised) with debugging symbols
set(CMAKE_BUILD_TYPE RELWITHDEBINFO)

# Disable CMake 3.12 policy that looks for <PackageName>_ROOT, since HDF5 does that automatically
if (POLICY CMP0074)
    cmake_policy(SET CMP0074 OLD)
endif ()

# Install and find HDF5
execute_process(WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/
                COMMAND bash install_hdf5.sh
                COMMAND tee /dev/tty
                COMMAND tail -n 1
                OUTPUT_VARIABLE _HDF5_ROOT)

set(ENV{HDF5_ROOT} "${PROJECT_SOURCE_DIR}/bin/CMake-hdf5-1.10.5/hdf5-1.10.5/")
message("HDF5_ROOT: $ENV{HDF5_ROOT}")

find_package(HDF5 COMPONENTS CXX REQUIRED)

link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS}
                    ./include/ )

# For non-standard library paths, provide Find<packagename> files in this dir.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable RDC and passing build flags to underlying compiler
set(CMAKE_CUDA_FLAGS "-rdc=true -std=c++11 --compiler-options -fPIC") 

find_package(CUDAToolkit)

add_subdirectory(src)

################################################################################

set(LIBS
    $<TARGET_OBJECTS:gpue_vortex>
    $<TARGET_OBJECTS:gpue_io>
    $<TARGET_OBJECTS:gpue_simulator>
    $<TARGET_OBJECTS:gpue_tests>)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIB "Build libgpue.so" OFF)
if(${BUILD_SHARED_LIB})
    add_library(gpue SHARED ${LIBS})
    target_link_libraries(gpue 
        CUDA::cufft
        CUDA::cudart
        ${HDF5_CXX_LIBRARIES}
    )
    set_target_properties(gpue PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# gpue
option(BUILD_BIN "Build gpue bninary" ON)
if(${BUILD_BIN})
    add_executable(gpue-exe ${CMAKE_SOURCE_DIR}/src/main.cu ${LIBS})
    target_link_libraries(gpue-exe 
        CUDA::cufft
        CUDA::cudart
        ${HDF5_CXX_LIBRARIES}
    )
    set_target_properties(gpue-exe PROPERTIES OUTPUT_NAME gpue)
    set_target_properties(gpue-exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
