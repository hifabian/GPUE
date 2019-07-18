project(gpue_io LANGUAGES CXX CUDA)

ADD_LIBRARY(gpue_io OBJECT fileIO.cu parser.cu)
target_include_directories(gpue_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/include )
