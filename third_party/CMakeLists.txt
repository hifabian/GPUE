cmake_minimum_required(VERSION 3.12)
project(gpue_third_party LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
if(${ENABLE_PYTHON})
    add_subdirectory(pybind11)  

    get_target_property(modifacecopts module INTERFACE_COMPILE_OPTIONS)
    list(REMOVE_ITEM modifacecopts "-fvisibility=hidden")
    set_target_properties(module PROPERTIES INTERFACE_COMPILE_OPTIONS "${modifacecopts}")
endif()

###############################################################################

