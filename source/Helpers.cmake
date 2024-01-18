
FUNCTION( COMPILE_PTX )
  set(options "")
  set(oneValueArgs TARGET_PATH)  
  set(multiValueArgs OPTIONS SOURCES INCLUDE)
  CMAKE_PARSE_ARGUMENTS( _FUNCTION "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  
  set( DEBUG_PTX OFF CACHE BOOL "Enable CUDA debugging")   
  if ( DEBUG_PTX )
	 set ( DEBUG_FLAGS "-G")
  else()
	 set ( DEBUG_FLAGS "")
  endif()
  
  set( LINE_GENERATE OFF CACHE BOOL "Generate line information in ptx files") 
  
  
	message(STATUS "_FUNCTION_INCLUDE: ${_FUNCTION_INCLUDE}")
  FOREACH(INCLUDE_DIR ${_FUNCTION_INCLUDE})
	  set ( INCL ${INCL} "-I\"${INCLUDE_DIR}\"" )  
  ENDFOREACH()
  
	message(STATUS "Compiling for architecture ${ARCH}")

	file ( MAKE_DIRECTORY "${_FUNCTION_TARGET_PATH}" )
	string (REPLACE ";" " " _FUNCTION_OPTIONS "${_FUNCTION_OPTIONS}")  
	separate_arguments( _OPTS WINDOWS_COMMAND "${_FUNCTION_OPTIONS}" )
	message ( STATUS "NVCC Options: ${_FUNCTION_OPTIONS}" )  
	message ( STATUS "NVCC Include: ${_FUNCTION_INCLUDE}" )

	set(_COMPILE_TYPE --ptx)

	if ( LINE_GENERATE )
		set(_OPTS ${_OPTS} -lineinfo)
	endif()

	FOREACH( input ${_FUNCTION_SOURCES} )
		get_filename_component( input_ext ${input} EXT )									
		get_filename_component( input_without_ext ${input} NAME_WE )						
		if ( ${input_ext} STREQUAL ".cu" )			
			set(_EXT .ptx)
			set( output "${input_without_ext}.ptx" )
			set( embedded_file "${input_without_ext}_embed.c")				
			set( output_with_path "${_FUNCTION_TARGET_PATH}/${input_without_ext}${_EXT}" )	
							
			message( STATUS "NVCC Compile: $${CUDA_NVCC_EXECUTABLE} ${MACHINE} ${_COMPILE_TYPE} ${DEBUG_FLAGS} ${OPT_FLAGS} ${_OPTS} ${input} ${INCL} -o ${output_with_path}")
			add_custom_command(
				OUTPUT  ${output_with_path}
				MAIN_DEPENDENCY ${input}
				DEPENDS ${_FILE_DEPENDENCY}
				COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} ${_COMPILE_TYPE} ${DEBUG_FLAGS} ${OPT_FLAGS} ${_OPTS} ${input} ${INCL} -o ${output_with_path} 
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			)
			
		endif()
  ENDFOREACH()

ENDFUNCTION()


