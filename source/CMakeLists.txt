cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(VPT LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
include(${CMAKE_SOURCE_DIR}/Helpers.cmake)
##################################################################

## Set output directories

set(_VCPKG_INCLUDE_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/")
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

##Set env mapping renders variable

set( RENDER_ENV_TEXTURES OFF CACHE BOOL "Render env sampling textures after creation")  
  if ( RENDER_ENV_TEXTURES )
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=1)
  else()
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=0)
  endif()
  
## Set ASSET_PATH variable to use with VolumeGVDB add path function
get_filename_component(ASSET_DIR ${CMAKE_SOURCE_DIR}/../assets ABSOLUTE)
add_definitions(-DASSET_PATH="${ASSET_DIR}/")
###################################################################

file( GLOB	HEADERS 
	"*.h"
	"${CMAKE_SOURCE_DIR}/gpu_vdb/*.h")

file ( GLOB COMMON "${CMAKE_SOURCE_DIR}/common/*.h"	)

file( GLOB CUDA_SOURCES 
	"*.cu"
	"*.cuh")
	
file( GLOB SOURCE_CPP
	"*.cpp"
	"${CMAKE_SOURCE_DIR}/gpu_vdb/*.cpp")
	
file( GLOB IMGUI_IMPL_SOURCES 
	"${CMAKE_SOURCE_DIR}/imgui/*.h"
	"${CMAKE_SOURCE_DIR}/imgui/*.cpp")
	
set(SOURCE_FILES
	${HEADERS}
	${CUDA_SOURCES}
	${SOURCE_CPP}
	${IMGUI_IMPL_SOURCES}
	${COMMON})

source_group(Cuda FILES ${CUDA_SOURCES})
source_group(Imgui FILES ${IMGUI_IMPL_SOURCES})
source_group(Helpers FILES ${COMMON})

## Add custom build command for cuda files 

find_program(CUDA_NVCC_EXECUTABLE nvcc)
message ( STATUS "Build CUDA kernels: ${CUDA_SOURCES}" )
_COMPILEPTX ( SOURCES ${CUDA_SOURCES} 
	TARGET_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} 
	GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS 
	INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${_VCPKG_INCLUDE_DIR},${CMAKE_CURRENT_SOURCE_DIR}/common,${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb" 
	OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128)

###################################################################

###################################################################
add_executable(vpt ${SOURCE_FILES})
target_link_libraries(vpt PRIVATE glfw imgui::imgui nlohmann_json::nlohmann_json cudart cuda openvdb)

include_directories(${_VCPKG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb)
