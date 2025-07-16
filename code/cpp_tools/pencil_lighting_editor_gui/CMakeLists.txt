project( pencil_lighting_editor_gui )

include( CMakeSourceFiles.txt )
add_executable( ${PROJECT_NAME} ${Headers} ${Sources} )

target_include_directories( ${PROJECT_NAME} PRIVATE "../../../external/imgui" )
target_include_directories( ${PROJECT_NAME} PRIVATE "../../../external/hdf5util" )
target_link_libraries( ${PROJECT_NAME} imgui HDF5 image strokes )

find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR} )

find_package( HDF5 REQUIRED )
if( NOT ${HDF5_FOUND} )
  message( FATAL_ERROR "HDF5 not found." )
endif()
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS} )
target_compile_definitions( ${PROJECT_NAME} PUBLIC USE_HDF5 )
target_link_libraries( ${PROJECT_NAME} ${HDF5_LIBRARIES} )

find_package( glm REQUIRED )
target_link_libraries( ${PROJECT_NAME} glm::glm )

find_package( PNG REQUIRED )
target_link_libraries( ${PROJECT_NAME} ${PNG_LIBRARIES} )
include_directories( ${PROJECT_NAME} ${PNG_INCLUDE_DIR} )

find_package( GLEW REQUIRED )
target_link_libraries( ${PROJECT_NAME} GLEW::GLEW )
find_package( glfw3 3.3 REQUIRED )
target_link_libraries( ${PROJECT_NAME} glfw )
find_package( OpenGL REQUIRED )
target_link_libraries( ${PROJECT_NAME} OpenGL::GL )
