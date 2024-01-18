
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(glfw3 CONFIG REQUIRED)
find_package(glew CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(OpenVDB CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)
find_package(tinyexr CONFIG REQUIRED)
find_package(tbb CONFIG REQUIRED)
find_package(OpenImageDenoise CONFIG REQUIRED)

include(${CMAKE_CURRENT_SOURCE_DIR}/Helpers.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/CompileObj.cmake)

##################################################################

# set external directories 

set(THIRDPARTY_DIR "${CMAKE_SOURCE_DIR}/thirdparty")

##Set env mapping renders variable

set( RENDER_ENV_TEXTURES OFF CACHE BOOL "Render env sampling textures after creation")  
  if ( RENDER_ENV_TEXTURES )
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=1)
  else()
	 add_definitions(-DRENDER_ENV_SAMPLE_TEXTURES=0)
  endif()
  
## Set ASSET_PATH variable to assets folder and copy the folder to binary directory

file( GLOB ASSET_FILES "${CMAKE_SOURCE_DIR}/assets/*.*")
file( INSTALL ${ASSET_FILES} DESTINATION "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/assets")

set(ASSET_DIR "./assets")
add_definitions(-DASSET_PATH="${ASSET_DIR}/")

###################################################################


## Create example instance file 

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
file(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fireball.ins" 
	"1\n"
	"${ASSET_DIR}/fireball.vdb\n"
	"3\n"
	"0 200 0 0 0 0 1 20\n"
	"200 200 0 0 0 0 1 20\n"
	"-200 200 0 0 0 0 1 20") 


## Create a cmd file to test vpt 

file(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/test_vpt.cmd"
	"echo on\n"
	"vpt.exe fireball.ins")


## Grab source files 

file( GLOB ATMOSPHERE
		"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.h"
		"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.cpp")


file( GLOB	HEADERS 
	"*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb/*.h")

file ( GLOB COMMON 
	"${CMAKE_CURRENT_SOURCE_DIR}/common/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"	)

file ( GLOB UTIL 
	"${CMAKE_CURRENT_SOURCE_DIR}/util/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/util/*.cpp"	)

file ( GLOB GEOMETRY 
	"${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.cpp"	)

file ( GLOB BVH 
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.h"	
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.cpp"	
	)

file ( GLOB CUDA_FUNCTIONS 
	"${CMAKE_CURRENT_SOURCE_DIR}/bvh/*.cu"	
	)

file( GLOB CUDA_SOURCES 
	"*.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/atmosphere/*.cu"
	"${CMAKE_CURRENT_SOURCE_DIR}/geometry/*.cu")
	
file( GLOB SOURCE_CPP
	"*.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb/*.cpp")
	
file( GLOB IMGUI_IMPL_SOURCES 
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.h"
	"${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.cpp")
	
set(SOURCE_FILES
	${HEADERS}
	${CUDA_SOURCES}
	${CUDA_FUNCTIONS}
	${SOURCE_CPP}
	${BVH}
	${IMGUI_IMPL_SOURCES}
	${COMMON}
	${ATMOSPHERE}
	${GEOMETRY}
	${UTIL})

## Group source files 

source_group(Cuda FILES ${CUDA_SOURCES})
source_group(bvh FILES ${BVH} ${CUDA_FUNCTIONS})
source_group(Imgui FILES ${IMGUI_IMPL_SOURCES})
source_group(Helpers FILES ${COMMON} ${UTIL})
source_group(Atmosphere FILES ${ATMOSPHERE})
source_group(Geometry FILES ${GEOMETRY})

## Add custom build command for cuda files 

find_program(CUDA_NVCC_EXECUTABLE nvcc)
message ( STATUS "Build CUDA kernels: ${CUDA_SOURCES}" )
COMPILE_PTX ( SOURCES ${CUDA_SOURCES} 
	TARGET_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} 
	INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} ${_VCPKG_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common ${THIRDPARTY_DIR}/cuda-noise/include ${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb
	OPTIONS -arch=compute_61 -code=sm_61 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128)


COMPILE_OBJ(
	SOURCES ${CUDA_FUNCTIONS}
	TYPE ${CMAKE_BUILD_TYPE}
	TARGET_PATH ${CMAKE_BINARY_DIR}/lib
	GENPATHS GEN_CUDA_FUNC_OBJS
	INCLUDE ${CMAKE_CURRENT_SOURCE_DIR} ${_VCPKG_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/common ${THIRDPARTY_DIR}/cuda-noise/include ${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb
)
message(STATUS "Genrated cuda functions objects ${GEN_CUDA_FUNC_OBJS}")

###################################################################

###################################################################
add_executable(vpt ${SOURCE_FILES})
target_link_libraries(vpt PRIVATE glfw GLEW::GLEW imgui::imgui cudart cuda OpenImageDenoise
			TBB::tbb OpenVDB::openvdb opengl32 OpenImageIO::OpenImageIO ${OIDN_LIBRARIES} ${_VCPKG_LIB_DIR}/tinyexr.lib ${GEN_CUDA_FUNC_OBJS})

set_target_properties(vpt PROPERTIES VS_GLOBAL_VcpkgEnabled true)
set_target_properties(vpt PROPERTIES VS_GLOBAL_VcpkgEnableManifest true)
target_compile_features(vpt PUBLIC cxx_std_17)

set_target_properties(vpt PROPERTIES CUDA_ARCHITECTURES "61;75;86")

include_directories(${_VCPKG_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/atmosphere)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/util)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/gpu_vdb)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/bvh)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/geometry)
include_directories(${THIRDPARTY_DIR}/cuda-noise/include)
include_directories(${THIRDPARTY_DIR}/OpenImageDenoise/include)
