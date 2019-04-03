cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)





#------------------------------------ CROSS-PLATFORM PTX COMPILE 
#
# _COMPILEPTX( SOURCES file1.cu file2.cu TARGET_PATH <path where ptxs should be stored> GENERATED_FILES ptx_sources NVCC_OPTIONS -arch=sm_20)
# Generates ptx files for the given source files. ptx_sources will contain the list of generated files.
#
FUNCTION( _COMPILEPTX )
  set(options "")
  set(oneValueArgs TARGET_PATH GENERATED GENPATHS INCLUDE)  
  set(multiValueArgs OPTIONS SOURCES)
  CMAKE_PARSE_ARGUMENTS( _COMPILEPTX "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  unset ( PTX_FILES CACHE )
  unset ( PTX_FILES_PATH CACHE )

  set( USE_DEBUG_PTX OFF CACHE BOOL "Enable CUDA debugging with NSight")  
  if ( USE_DEBUG_PTX )
	 set ( DEBUG_FLAGS ";-g;-G;-D_DEBUG;-DEBUG")
  else()
	 set ( DEBUG_FLAGS "")
  endif()
  
  if ( WIN32 ) 
		# Windows - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}/Debug" )
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}/Release" )
		string (REPLACE ";" " " _COMPILEPTX_OPTIONS "${_COMPILEPTX_OPTIONS}")  
		separate_arguments( _OPTS WINDOWS_COMMAND "${_COMPILEPTX_OPTIONS}" )
		message ( STATUS "NVCC Options: ${_COMPILEPTX_OPTIONS}" )  
		message ( STATUS "NVCC Include: ${_COMPILEPTX_INCLUDE}" )

        set ( INCL "-I\"${_COMPILEPTX_INCLUDE}\"" )

		# Custom build rule to generate ptx files from cuda files
		FOREACH( input ${_COMPILEPTX_SOURCES} )
			get_filename_component( input_ext ${input} EXT )									# Input extension
			get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
			if ( ${input_ext} STREQUAL ".cu" )			
				
				# Set output names
				set( output "${input_without_ext}.ptx" )							# Output name
				set( output_with_path "${_COMPILEPTX_TARGET_PATH}/$(Configuration)/${input_without_ext}.ptx" )	# Output with path
				set( output_with_quote "\"${output_with_path}\"" )
				LIST( APPEND PTX_FILES ${output} )		# Append to output list
				LIST( APPEND PTX_FILES_PATH ${output_with_path} )
    
				message( STATUS "NVCC Compile: ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_COMPILEPTX_OPTIONS} ${input} ${INCL} -o ${output_with_path} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
    
				add_custom_command(
					OUTPUT  ${output_with_path}
					MAIN_DEPENDENCY ${input}
					COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --ptx ${_OPTS} ${input} ${INCL} -o ${output_with_quote} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
				)			
			endif()
		ENDFOREACH( )
  else ()
		# Linux - PTX compile
		file ( MAKE_DIRECTORY "${_COMPILEPTX_TARGET_PATH}" )
    FOREACH(input ${_COMPILEPTX_SOURCES})
      get_filename_component( input_ext ${input} EXT )									# Input extension
      get_filename_component( input_without_ext ${input} NAME_WE )						# Input base
      if ( ${input_ext} STREQUAL ".cu" )			
        # Set output names
        set( output "${input_without_ext}.ptx" ) # Output name
        set( output_with_path "${_COMPILEPTX_TARGET_PATH}/${input_without_ext}.ptx" )	# Output with path

        set( compile_target_ptx "${input_without_ext}_PTX")
        set( custom_command_var "${input_without_ext}_OUTPUT")
        # compile ptx
        cuda_compile_ptx(custom_command_var ${input} OPTIONS "${DEBUG_FLAGS}")
        # This will only configure file generation, we need to add a target to
        # generate a file cuda_generated_<counter>_${input_without_ext}.ptx
        # Add custom command to rename to simply ${input_without_ext}.ptx
        add_custom_command(OUTPUT ${output_with_path}
                          COMMAND ${CMAKE_COMMAND} -E rename ${custom_command_var} ${output_with_path}
                          DEPENDS ${custom_command_var})
        add_custom_target(${compile_target_ptx} ALL DEPENDS ${input} ${output_with_path} SOURCES ${input})

        # Add this output file to list of generated ptx files
        LIST(APPEND PTX_FILES ${output})
        LIST(APPEND PTX_FILES_PATH ${output_with_path} )
      endif()
    ENDFOREACH()
  endif()

  set( ${_COMPILEPTX_GENERATED} ${PTX_FILES} PARENT_SCOPE)
  set( ${_COMPILEPTX_GENPATHS} ${PTX_FILES_PATH} PARENT_SCOPE)

ENDFUNCTION()



project(GVDB_PT LANGUAGES CXX CUDA)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(glfw3 REQUIRED)

get_filename_component(GVDB_DIR ${CMAKE_SOURCE_DIR}/../gvdb ABSOLUTE)

# Set output directories

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/GVDB_PT/bin)

if ( NOT DEFINED ASSET_PATH ) 
  get_filename_component ( _assets "${GVDB_DIR}/shared_assets" REALPATH )
  set ( ASSET_PATH ${_assets} CACHE PATH "Full path to gvdb/shared_assets/" )  
endif()
add_definitions(-DASSET_PATH="${ASSET_PATH}/")


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

#file(GLOB CUDA_FILES RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" *.cu *.cuh)
#message ( STATUS "Build CUDA kernels: ${CUDA_FILES}" )
#_COMPILEPTX ( SOURCES ${CUDA_FILES} TARGET_PATH ${GVDB_DIR}/bin GENERATED CUDA_PTX GENPATHS CUDA_PTX_PATHS INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}" OPTIONS -arch=compute_30 -code=sm_30 --ptxas-options=-v -O3 --use_fast_math --maxrregcount=128 )


find_file(LIB_GVDB NAME "libgvdb.lib" HINTS ${GVDB_DIR}/lib)
message(STATUS "libgvdb path: " ${LIB_GVDB})

target_link_libraries(gvdb_pt PRIVATE glfw ${LIB_GVDB})

target_include_directories(gvdb_pt PRIVATE "${GVDB_DIR}/include")
target_include_directories(gvdb_pt PRIVATE "${GVDB_DIR}/sample_utils")

set(INSTALL_LIST
	${PTX}
	${DLLS}
	${LIBS})

add_custom_command(TARGET gvdb_pt POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy  ${INSTALL_LIST} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )