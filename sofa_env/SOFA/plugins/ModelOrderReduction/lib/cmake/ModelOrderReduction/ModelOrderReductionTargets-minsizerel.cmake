#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ModelOrderReduction" for configuration "MinSizeRel"
set_property(TARGET ModelOrderReduction APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(ModelOrderReduction PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libModelOrderReduction.so.1.0"
  IMPORTED_SONAME_MINSIZEREL "libModelOrderReduction.so.1.0"
  )

list(APPEND _IMPORT_CHECK_TARGETS ModelOrderReduction )
list(APPEND _IMPORT_CHECK_FILES_FOR_ModelOrderReduction "${_IMPORT_PREFIX}/lib/libModelOrderReduction.so.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
