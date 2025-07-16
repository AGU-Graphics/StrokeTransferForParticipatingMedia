project( HDF5 )

include( CMakeSourceFiles.txt )
add_library( ${PROJECT_NAME} ${Headers} ${Sources} )

find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} )

find_package( HDF5 REQUIRED )
if( NOT ${HDF5_FOUND} )
  message( FATAL_ERROR "HDF5 not found." )
endif()
target_include_directories( ${PROJECT_NAME} PRIVATE ${HDF5_INCLUDE_DIRS} )
target_compile_definitions( ${PROJECT_NAME} PRIVATE USE_HDF5 )
target_link_libraries( ${PROJECT_NAME} PRIVATE ${HDF5_LIBRARIES} )
