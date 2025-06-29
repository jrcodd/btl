# CMake package configuration file for the plugin 'CImgPlugin'

### Expanded from @PACKAGE_GUARD@ by SofaMacrosInstall.cmake ###
include_guard()
list(APPEND CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../bin")
list(APPEND CMAKE_LIBRARY_PATH "${CMAKE_CURRENT_LIST_DIR}/../../../lib")
################################################################

####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CImgPluginConfig.cmake.in                            ########

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

set(CIMGPLUGIN_HAVE_JPEG 1)
set(CIMGPLUGIN_HAVE_TIFF 1)
set(CIMGPLUGIN_HAVE_PNG 1)
set(CIMGPLUGIN_HAVE_ZLIB 1)

find_package(Sofa.Helper QUIET REQUIRED)
find_package(Sofa.DefaultType QUIET REQUIRED)

set(CImg_INCLUDE_DIRS "/builds/workspace/sofa-custom/refs/heads/v24.06/ubuntu_gcc_release_full_python3.10/build/_deps/cimg-src")
find_package(CImg QUIET REQUIRED)

if(CIMGPLUGIN_HAVE_JPEG)
    find_package(JPEG QUIET REQUIRED)
endif()
if(CIMGPLUGIN_HAVE_TIFF)
    find_package(TIFF QUIET REQUIRED)
endif()
if(CIMGPLUGIN_HAVE_PNG)
    find_package(PNG QUIET REQUIRED)
endif()
if(CIMGPLUGIN_HAVE_ZLIB)
    find_package(ZLIB QUIET REQUIRED)
endif()

if(NOT TARGET CImgPlugin)
    include("${CMAKE_CURRENT_LIST_DIR}/CImgPluginTargets.cmake")
endif()

check_required_components(CImgPlugin)
