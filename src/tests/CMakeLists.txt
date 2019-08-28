project(gpue_tests LANGUAGES CUDA)
add_library(gpue_tests OBJECT unit_test.cu)
set_property(TARGET gpue_tests PROPERTY COMPILE_FLAGS "-dc")

#enable_testing()
if(${CMAKE_TESTING_ENABLED})
    #add_executable(test_gpue_vortices VtxTest.cu)
    #target_link_libraries(test_gpue_vortices gpue_vortices)

    #add_executable(test_gpue_lattice graphtest.cu)
    #target_link_libraries(test_gpue_lattice gpue_lattice gpue_vortices)
endif()
