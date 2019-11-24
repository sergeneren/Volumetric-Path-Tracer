cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(VPT LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


##################################################################

## Set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/vpt/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/vpt/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/vpt/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/vpt/bin)

##################################################################

get_filename_component(GVDB_DIR ${CMAKE_SOURCE_DIR}/thirdparty/gvdb-voxels/_output ABSOLUTE)

add_subdirectory(${CMAKE_SOURCE_DIR}/source)