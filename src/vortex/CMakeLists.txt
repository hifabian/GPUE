project(gpue_vortex LANGUAGES CUDA)

ADD_LIBRARY(gpue_vortex OBJECT edge.cu node.cu lattice.cu tracker.cu vort.cu vortex_3d.cu)
target_include_directories(gpue_vortex PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include )

#enable_testing()
if(${CMAKE_TESTING_ENABLED})
    #add_executable(test_gpue_vortices VtxTest.cu)
    #target_link_libraries(test_gpue_vortices gpue_vortex)

    #add_executable(test_gpue_lattice graphtest.cu)
    #target_link_libraries(test_gpue_lattice gpue_vortex)
endif()
