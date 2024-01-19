#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OpenImageDenoise_core" for configuration "Release"
set_property(TARGET OpenImageDenoise_core APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenImageDenoise_core PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/OpenImageDenoise_core.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/OpenImageDenoise_core.dll"
  )

list(APPEND _cmake_import_check_targets OpenImageDenoise_core )
list(APPEND _cmake_import_check_files_for_OpenImageDenoise_core "${_IMPORT_PREFIX}/lib/OpenImageDenoise_core.lib" "${_IMPORT_PREFIX}/bin/OpenImageDenoise_core.dll" )

# Import target "OpenImageDenoise" for configuration "Release"
set_property(TARGET OpenImageDenoise APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(OpenImageDenoise PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/OpenImageDenoise.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "OpenImageDenoise_core"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/OpenImageDenoise.dll"
  )

list(APPEND _cmake_import_check_targets OpenImageDenoise )
list(APPEND _cmake_import_check_files_for_OpenImageDenoise "${_IMPORT_PREFIX}/lib/OpenImageDenoise.lib" "${_IMPORT_PREFIX}/bin/OpenImageDenoise.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
