#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Sofa.SimpleApi" for configuration "MinSizeRel"
set_property(TARGET Sofa.SimpleApi APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Sofa.SimpleApi PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofa.SimpleApi.so.24.06.00"
  IMPORTED_SONAME_MINSIZEREL "libSofa.SimpleApi.so.24.06.00"
  )

list(APPEND _IMPORT_CHECK_TARGETS Sofa.SimpleApi )
list(APPEND _IMPORT_CHECK_FILES_FOR_Sofa.SimpleApi "${_IMPORT_PREFIX}/lib/libSofa.SimpleApi.so.24.06.00" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
