
#------------------------------------ CROSS-PLATFORM OBJ COMPILE

FUNCTION( COMPILE_OBJ )
  set(options "")
  set(oneValueArgs TARGET_PATH TYPE GENERATED GENPATHS)  
  set(multiValueArgs OPTIONS SOURCES ARCHS INCLUDE)
  CMAKE_PARSE_ARGUMENTS( _FUNCTION "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  # Match the bitness of the ptx to the bitness of the application
  set( MACHINE "--machine=32" )
  if( CMAKE_SIZEOF_VOID_P EQUAL 8)
    set( MACHINE "--machine=64" )
  endif()
  unset ( CUBIN_FILES CACHE )
  unset ( CUBIN_FILES_PATH CACHE )
  unset ( EMBEDDED_FILES CACHE )
    
  FOREACH(INCLUDE_DIR ${_FUNCTION_INCLUDE})
	  set ( INCL ${INCL} "-I\"${INCLUDE_DIR}\"" )  
  ENDFOREACH()
  
	set ( ARCHS -arch=sm_61 )  
  
	message(STATUS "Compiling for architecture ${ARCH}")
	
	file ( MAKE_DIRECTORY "${_FUNCTION_TARGET_PATH}" )
	string (REPLACE ";" " " _FUNCTION_OPTIONS "${_FUNCTION_OPTIONS}")  
	separate_arguments( _OPTS WINDOWS_COMMAND "${_FUNCTION_OPTIONS}" )
	message ( STATUS "NVCC Options: ${_FUNCTION_OPTIONS}" )  
	message ( STATUS "NVCC Include: ${_FUNCTION_INCLUDE}" )

	#Set initial options
	set(_OPTS ${_OPTS} -Xcompiler "/EHsc,/Od,/Zi,/RTC1" ${ARCHS})
	
	#Set debug or relase linking 
	if(_FUNCTION_TYPE MATCHES "Debug")
		set(_OPTS ${_OPTS} -Xcompiler "/MDd")
	else()
		set(_OPTS ${_OPTS} -Xcompiler "/MD")
	endif()
	
	set(_OPTS ${_OPTS} -w)
	
	FOREACH( input ${_FUNCTION_SOURCES} )
		get_filename_component( input_ext ${input} EXT )									
		get_filename_component( input_without_ext ${input} NAME_WE )						
		if ( ${input_ext} STREQUAL ".cu" )			
			set(_EXT .lib)
			
			set( output "${input_without_ext}.lib" )
			set( output_with_path "${_FUNCTION_TARGET_PATH}/${input_without_ext}${_EXT}" )	
			set( output_with_quote "\"${output_with_path}\"" )
			LIST( APPEND OBJ_FILES ${output} )		
			LIST( APPEND OBJ_FILES_PATH ${output_with_path} )				
							
			message( STATUS "NVCC Compile: ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --lib -cudart static ${DEBUG_FLAGS} ${_OPTS} ${input} -o ${output_with_path} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}")
			add_custom_command(
				OUTPUT  ${output_with_path}
				MAIN_DEPENDENCY ${input}
				DEPENDS ${_FILE_DEPENDENCY}
				COMMAND ${CUDA_NVCC_EXECUTABLE} ${MACHINE} --compile ${DEBUG_FLAGS} ${OPT_FLAGS} ${_OPTS} ${input} ${INCL} -o ${output_with_quote} 
				WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			)
			
		endif()
	ENDFOREACH( )

  set( ${_FUNCTION_GENERATED} ${OBJ_FILES} PARENT_SCOPE)
  set( ${_FUNCTION_GENPATHS} ${OBJ_FILES_PATH} PARENT_SCOPE)

ENDFUNCTION()
