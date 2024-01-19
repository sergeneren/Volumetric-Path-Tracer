## Copyright 2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was Config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(OIDN_DEVICE_CPU  ON)
set(OIDN_DEVICE_SYCL OFF)
set(OIDN_DEVICE_CUDA ON)
set(OIDN_DEVICE_HIP  OFF)

set(OIDN_FILTER_RT ON)
set(OIDN_FILTER_RTLIGHTMAP ON)

set(OIDN_STATIC_LIB OFF)

if(OIDN_STATIC_LIB AND OIDN_DEVICE_CPU)
  include(CMakeFindDependencyMacro)
  find_dependency(TBB)
endif()

include("${CMAKE_CURRENT_LIST_DIR}/OpenImageDenoiseTargets.cmake")

check_required_components(OpenImageDenoise)
