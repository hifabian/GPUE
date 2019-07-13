project(gpue_vortex_lib LANGUAGES CUDA)

add_library(gpue_lattice STATIC edge.cu node.cu lattice.cu)
target_include_directories(gpue_lattice PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include )

add_library(gpue_vortices STATIC tracker.cu vort.cu vortex_3d.cu)
target_include_directories(gpue_vortices PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include )

#enable_testing()
if(${CMAKE_TESTING_ENABLED})
    #add_executable(test_gpue_vortices VtxTest.cu)
    #target_link_libraries(test_gpue_vortices gpue_vortices)

    #add_executable(test_gpue_lattice graphtest.cu)
    #target_link_libraries(test_gpue_lattice gpue_lattice gpue_vortices)
endif()
