#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Sofa.Component.Constraint.Lagrangian.Correction" for configuration "MinSizeRel"
set_property(TARGET Sofa.Component.Constraint.Lagrangian.Correction APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(Sofa.Component.Constraint.Lagrangian.Correction PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libSofa.Component.Constraint.Lagrangian.Correction.so.24.06.00"
  IMPORTED_SONAME_MINSIZEREL "libSofa.Component.Constraint.Lagrangian.Correction.so.24.06.00"
  )

list(APPEND _IMPORT_CHECK_TARGETS Sofa.Component.Constraint.Lagrangian.Correction )
list(APPEND _IMPORT_CHECK_FILES_FOR_Sofa.Component.Constraint.Lagrangian.Correction "${_IMPORT_PREFIX}/lib/libSofa.Component.Constraint.Lagrangian.Correction.so.24.06.00" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
