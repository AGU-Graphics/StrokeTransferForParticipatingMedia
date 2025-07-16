project( image )

include( CMakeSourceFiles.txt )
add_library( ${PROJECT_NAME} ${Headers} ${Sources} )

find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ${PROJECT_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} )

find_package( PNG REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${PNG_LIBRARIES} )
include_directories( ${PROJECT_NAME} ${PNG_INCLUDE_DIR} )
