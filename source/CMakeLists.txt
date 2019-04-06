cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
project(GVDB_PT LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(glfw3 REQUIRED)
find_package(imgui CONFIG REQUIRED)

## Set gvdb library 
get_filename_component(GVDB_DIR ${CMAKE_SOURCE_DIR}/../thirdparty/gvdb ABSOLUTE)
get_filename_component(STB_DIR ${CMAKE_SOURCE_DIR}/../thirdparty/stb_image ABSOLUTE)

include(${GVDB_DIR}/sample_utils/Helpers.cmake)
##################################################################

## Set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

##################################################################


## Set ASSET_PATH variable to use with VolumeGVDB add path function
get_filename_component(ASSET_DIR ${CMAKE_SOURCE_DIR}/../assets ABSOLUTE)
add_definitions(-DASSET_PATH="${ASSET_DIR}/")
###################################################################

file( GLOB	DLLS 
		"${GVDB_DIR}/bin/*.dll")
file( GLOB	LIBS 
		"${GVDB_DIR}/bin/*.lib")
file( GLOB	PTX 
		"${GVDB_DIR}/bin/*.ptx")

file( GLOB	HEADERS "*.h")

file( GLOB CUDA_SOURCES 
	"*.cu"
	"*.cuh")
	
file( GLOB SOURCE_CPP
	"*.cpp")
	
set(SOURCE_FILES
	${HEADERS}
	${CUDA_SOURCES}
	${SOURCE_CPP}
	)

source_group(Cuda FILES ${CUDA_SOURCES})

add_executable(gvdb_pt ${SOURCE_FILES})

## Add custom build command for cuda files 

file(GLOB CUDA_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" *.cu *.cuh)
find_program(CUDA_NVCC_EXECUTABLE nvcc)
message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )
_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR},${GVDB_DIR}/include" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 )

###################################################################

## Find gvdb library 

find_file(LIB_GVDB NAME "libgvdb.lib" HINTS ${GVDB_DIR}/lib)
message(STATUS "libgvdb path: " ${LIB_GVDB})

###################################################################

target_link_libraries(gvdb_pt PRIVATE glfw imgui::imgui ${LIB_GVDB} cudart cuda )

target_include_directories(gvdb_pt PRIVATE "${GVDB_DIR}/include")
target_include_directories(gvdb_pt PRIVATE "${GVDB_DIR}/sample_utils")
target_include_directories(gvdb_pt PRIVATE "${GVDB_DIR}/sample_utils")

## Set dlls to be installed to executable directory after build 

set(INSTALL_LIST
	${PTX}
	${DLLS}
	${LIBS})

add_custom_command(TARGET gvdb_pt POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy  ${INSTALL_LIST} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )

###################################################################