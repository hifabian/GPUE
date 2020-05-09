cmake_minimum_required(VERSION 3.12)
project(gpue_python LANGUAGES CXX)

###############################################################################
# Include all nested sources directories
###############################################################################
set(GPUE_PYTHON_SUBDIRS    PyGPUE )

foreach(MOD ${GPUE_PYTHON_SUBDIRS})
    add_subdirectory(${MOD})
endforeach()

if(${ENABLE_PYTHON})
    # Install the bound module into the GPUE Python environment
    configure_file(setup.py ${CMAKE_CURRENT_BINARY_DIR}/setup.py )
    find_package (Python COMPONENTS Interpreter Development REQUIRED)
    add_custom_target(PipInstaller ALL DEPENDS _PyGPUE _PyGPUE_LG)
    add_custom_command(TARGET PipInstaller
        # Run after all other rules within the target have been executed
        POST_BUILD 
        DEPENDS Python::Interpreter
        COMMAND ${Python_EXECUTABLE} -m pip install -e . --user
        COMMENT "Installing Python bindings into local installation"
        VERBATIM
    )
endif()
