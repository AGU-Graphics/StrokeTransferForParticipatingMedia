project( render_surface_features_cli )

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
include(libigl)

# Enable the target igl::glfw
igl_include(glfw)

include( CMakeSourceFiles.txt )
add_executable( ${PROJECT_NAME} ${Headers} ${Sources} )

target_include_directories( ${PROJECT_NAME} PRIVATE "../../../external/hdf5util" )
target_link_libraries( ${PROJECT_NAME} HDF5 image strokes imgui )

find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR} )

find_package( HDF5 REQUIRED )
if( NOT ${HDF5_FOUND} )
  message( FATAL_ERROR "HDF5 not found." )
endif()
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS} )
target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_HDF5 )
target_link_libraries( ${PROJECT_NAME} ${HDF5_LIBRARIES} )

find_package( PNG REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${PNG_LIBRARIES} )
include_directories( ${PROJECT_NAME} ${PNG_INCLUDE_DIR} )

find_package( GLEW REQUIRED )
target_link_libraries( ${PROJECT_NAME} GLEW::GLEW )
find_package( glfw3 3.3 REQUIRED )
target_link_libraries( ${PROJECT_NAME} glfw )
find_package( OpenGL REQUIRED )
target_link_libraries( ${PROJECT_NAME} OpenGL::GL )

find_package( Boost REQUIRED COMPONENTS program_options json )
target_link_libraries( ${PROJECT_NAME} Boost::program_options Boost::json )

target_link_libraries( ${PROJECT_NAME} igl::glfw )
