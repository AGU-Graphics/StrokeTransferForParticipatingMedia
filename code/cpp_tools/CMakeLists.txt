cmake_minimum_required( VERSION 3.5.1...3.29.3 )
project( render_strokes_cpp )

################################################################################
### Configure and check the build setting options

# If the user provided a CMAKE_BUILD_TYPE, create a cache variable
if( NOT CMAKE_BUILD_TYPE STREQUAL "" )
  set( CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE )
  # Otherwise default to Release
else( NOT CMAKE_BUILD_TYPE STREQUAL "" )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel Coverage." FORCE )
endif( NOT CMAKE_BUILD_TYPE STREQUAL "" )

# Enforce the use of a valid build type
set( VALID_BUILD_TYPES None Debug Release RelWithDebInfo MinSizeRel Coverage )
if( NOT ";${VALID_BUILD_TYPES};" MATCHES ";${CMAKE_BUILD_TYPE};" )
  message( FATAL_ERROR "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel Coverage." )
endif()


################################################################################
### Set flags

# Disable compiler language extensions
set( CMAKE_CXX_EXTENSIONS OFF )
# Enable C++20
set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )

# Append an extra space if flags are already set
if( CMAKE_CXX_FLAGS )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} " )
endif( CMAKE_CXX_FLAGS )

if( CMAKE_C_FLAGS )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} " )
endif( CMAKE_C_FLAGS )

if( POLICY CMP0167 )
  cmake_policy( SET CMP0167 NEW )
endif()

################################################################################
### Set paths, etc

# Create a default location for custom CMake modules
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake" )
message( ${PROJECT_SOURCE_DIR}/cmake )

# Allow projects to reference source relative to the root of the project
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

################################################################################
### 

add_subdirectory( lib/image )
add_subdirectory( lib/imgui )
add_subdirectory( lib/HDF5 )
add_subdirectory( lib/strokes )
add_subdirectory( gen_strokes_cli )
add_subdirectory( oil_lighting_editor_gui )
add_subdirectory( pencil_lighting_editor_gui )
add_subdirectory( render_strokes_cli )
add_subdirectory( render_strokes_pencil_cli )
add_subdirectory( exr2hdf_cli )
add_subdirectory( render_surface_features_cli )