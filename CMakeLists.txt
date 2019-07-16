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
project(gpue LANGUAGES CXX CUDA)

###############################################################################

# Require build directory; forbid modifying the source files during compilation
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

###############################################################################

# Ensure C++11 enabled
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use release build type (performance optimised) with debugging symbols
set(CMAKE_BUILD_TYPE RELWITHDEBINFO)

# Enable building test suite.
option(BUILD_TESTS "Build GPUE tests" ON)
if (${BUILD_TESTS})
    enable_testing()
endif()

################################################################################

# For non-standard library paths, provide Find<packagename> files in this dir.
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

################################################################################

# Enable RDC and passing build flags to underlying compiler
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-rdc=true;-O3;-Xcompiler -std=c++11;")

################################################################################

add_subdirectory(src)
find_package(CUDA REQUIRED)

################################################################################

if(false)
# libgpue.so
ADD_LIBRARY(gpue SHARED)
target_link_libraries(gpue 
    gpue_vortex 
    gpue_io 
    gpue_simulator 
    gpue_tests
    ${CUDA_LIBRARIES} 
    ${CUDA_CUFFT_LIBRARIES}
)
endif()

# gpue
ADD_EXECUTABLE(gpue-exe ${CMAKE_SOURCE_DIR}/src/main.cu)
target_link_libraries(gpue-exe 
    gpue_vortex
    gpue_io
    gpue_simulator
    gpue_tests
    #gpue_utils
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)
set_target_properties(gpue-exe PROPERTIES OUTPUT_NAME gpue)
set_target_properties(gpue-exe PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
