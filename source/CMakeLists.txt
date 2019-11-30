
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
include(${CMAKE_CURRENT_SOURCE_DIR}/Helpers.cmake)

##################################################################

# set external directories 

set(_VCPKG_INCLUDE_DIR "${_VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/")
set(THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty/")


#set OIDN libraries 

set(OID_DIR "${THIRDPARTY_DIR}/OpenImageDenoise")
find_library(OIDN_LIB "OpenImageDenoise" "${OID_DIR}/lib")
find_library(OIDN_MKLDNN_LIB "mkldnn" "${OID_DIR}/lib")
find_library(OIDN_COMMON_LIB "common" "${OID_DIR}/lib")

set(OIDN_LIBRARIES ${OIDN_LIB} ${OIDN_MKLDNN_LIB} ${OIDN_COMMON_LIB})

##Set env mapping renders variable

set( RENDER_ENV_TEXTURES OFF CACHE BOOL "Render env sampling textures after creation")  
  if ( RENDER_ENV_TEXTURES )
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=1)
  else()
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=0)
  endif()
  
## Set ASSET_PATH variable to use with VolumeGVDB add path function
get_filename_component(ASSET_DIR ${CMAKE_SOURCE_DIR}/assets ABSOLUTE)
add_definitions(-DASSET_PATH="${ASSET_DIR}/")
###################################################################

file( GLOB ATMOSPHERE
		"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.cpp")


file( GLOB	HEADERS 
	"*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb/*.h")

file ( GLOB COMMON "${CMAKE_CURRENT_SOURCE_DIR}/common/*.h"	)

file ( GLOB BVH 
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.cpp"	
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.cu"	
	)

file( GLOB CUDA_SOURCES 
	"*.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.cu")
	
file( GLOB SOURCE_CPP
	"*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb/*.cpp")
	
file( GLOB IMGUI_IMPL_SOURCES 
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.cpp")
	
set(SOURCE_FILES
	${HEADERS}
	${CUDA_SOURCES}
	${SOURCE_CPP}
	${BVH}
	${IMGUI_IMPL_SOURCES}
	${COMMON}
	${ATMOSPHERE})

source_group(Cuda FILES ${CUDA_SOURCES})
source_group(bvh FILES ${BVH})
source_group(Imgui FILES ${IMGUI_IMPL_SOURCES})
source_group(Helpers FILES ${COMMON})
source_group(Atmosphere FILES ${ATMOSPHERE})

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
target_link_libraries(vpt PRIVATE glfw imgui::imgui nlohmann_json::nlohmann_json cudart cuda openvdb ${OIDN_LIBRARIES})
target_compile_options(vpt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_30,code=sm_30>)

include_directories(${_VCPKG_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/atmosphere)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bvh)
include_directories(${OID_DIR}/include)
