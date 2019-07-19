################################################################################

cmake_minimum_required(VERSION 3.8)
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
cmake_policy(SET CMP0074 OLD)

# Install and find HDF5
execute_process(WORKING_DIRECTORY ./bin/
                COMMAND bash install_hdf5.sh
                COMMAND tee /dev/tty
                COMMAND tail -n 1
                OUTPUT_VARIABLE _HDF5_ROOT)

set(ENV{HDF5_ROOT} "./bin/CMake-hdf5-1.10.5/hdf5-1.10.5/")
message("HDF5_ROOT: $ENV{HDF5_ROOT}")

find_package(HDF5 COMPONENTS CXX REQUIRED)

link_directories( ${HDF5_LIBRARY_DIRS} )
include_directories( ${HDF5_INCLUDE_DIRS} )

# For non-standard library paths, provide Find<packagename> files in this dir.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Enable RDC and passing build flags to underlying compiler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-rdc=true;-O3;-Xcompiler -std=c++11;")

add_subdirectory(src)
find_package(CUDA REQUIRED)

# message("HDF5_LIBRARY_DIRS: ${HDF5_LIBRARY_DIRS}")
# message("HDF5_INCLUDE_DIRS: ${HDF5_INCLUDE_DIRS}")
# message("HDF5_CXX_LIBRARIES: ${HDF5_CXX_LIBRARIES}")
# message("CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
# message("CUDA_CUFFT_LIBRARIES: ${CUDA_CUFFT_LIBRARIES}")
# message("CUDA_cudadevrt_LIBRARY: ${CUDA_cudadevrt_LIBRARY}")

################################################################################

if(false)
# libgpue.so
add_library(gpue SHARED)
target_link_libraries(gpue 
    gpue_vortex 
    gpue_io 
    gpue_simulator 
    gpue_tests
    # gpue_utils
    ${CUDA_LIBRARIES} 
    ${CUDA_CUFFT_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)
endif()

# gpue
add_executable(gpue-exe ${CMAKE_SOURCE_DIR}/src/main.cu)
target_link_libraries(gpue-exe 
    gpue_vortex
    gpue_io
    gpue_simulator
    gpue_tests
    # gpue_utils
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
)

target_compile_features(gpue-exe PUBLIC cxx_std_11)
set_target_properties(gpue-exe PROPERTIES OUTPUT_NAME gpue)
set_target_properties(gpue-exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
