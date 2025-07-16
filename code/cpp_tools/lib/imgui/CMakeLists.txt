project( imgui )

include( CMakeSourceFiles.txt )
add_library( ${PROJECT_NAME} ${Headers} ${Sources} )

find_package( Eigen3 3.2.9 REQUIRED )
target_include_directories( ${PROJECT_NAME} SYSTEM PUBLIC ${EIGEN3_INCLUDE_DIR} )

find_package( glm REQUIRED )
target_link_libraries( ${PROJECT_NAME} glm::glm )

find_package( GLEW REQUIRED )
target_link_libraries( ${PROJECT_NAME} GLEW::GLEW )
find_package( glfw3 3.3 REQUIRED )
target_link_libraries( ${PROJECT_NAME} glfw )
find_package( OpenGL REQUIRED )
target_link_libraries( ${PROJECT_NAME} OpenGL::GL )
