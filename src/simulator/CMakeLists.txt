project(gpue_simulator LANGUAGES CUDA)

ADD_LIBRARY(gpue_simulator OBJECT 
    ds.cu 
    dynamic.cu
    evolution.cu
    init.cu
    kernels.cu
    manip.cu
    minions.cu
    operators.cu
    split_op.cu
#    wavefunction.cu
)
target_include_directories(gpue_simulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include )
set_property(TARGET gpue_simulator PROPERTY COMPILE_FLAGS "-dc")

# Required for separable compilation
target_link_libraries(gpue_simulator ${CUDA_cudadevrt_LIBRARY})

#enable_testing()
if(${CMAKE_TESTING_ENABLED})
    #add_executable(test_gpue_vortices VtxTest.cu)
    #target_link_libraries(test_gpue_vortices gpue_vortices)

    #add_executable(test_gpue_lattice graphtest.cu)
    #target_link_libraries(test_gpue_lattice gpue_lattice gpue_vortices)
endif()
