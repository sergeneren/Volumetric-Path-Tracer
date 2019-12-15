cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(VPT LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(_VCPKG_INCLUDE_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/")

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

option (BUILD_VPT "Builds Volumetric Path Tracer" ON)
option (BUILD_INSTANCER "Builds Houdini plugin to save instance files" OFF)

set(DebugLevel "Error" CACHE STRING "Choose a debug level that VPT will use in message display")
set_property(CACHE DebugLevel PROPERTY STRINGS Error Warning Log)

message(STATUS "Debug Level is ${DebugLevel}")
if(${DebugLevel} STREQUAL "Error")
	add_definitions(-DDEBUG_LEVEL_ERROR)
elseif(${DebugLevel} STREQUAL "Warning")
	add_definitions(-DDEBUG_LEVEL_WARNING)
elseif(${DebugLevel} STREQUAL "Log")
	add_definitions(-DDEBUG_LEVEL_LOG)
endif()



if(BUILD_VPT)
	add_subdirectory(${CMAKE_SOURCE_DIR}/source)
endif(BUILD_VPT)

if(BUILD_INSTANCER)
	add_subdirectory(${CMAKE_SOURCE_DIR}/instancer_hda)
endif(BUILD_INSTANCER)