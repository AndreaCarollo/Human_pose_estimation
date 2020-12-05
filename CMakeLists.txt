cmake_minimum_required(VERSION 3.5)

# Source path:
set(SOURCE_DIR ${CMAKE_HOME_DIRECTORY}/src)
# List of all files in src with name ending in '.cpp':
file(GLOB LIB_SOURCES ${SOURCE_DIR}/lib/*.cpp)


# TARGETS
# static library containing everything in src
#     first argument is the name of the target
#     second argument is the type of the library (STATIC of SHARED)
#     third argument is a list of source files
add_library(followmelib STATIC ${LIB_SOURCES})

set( CMAKE_CXX_STANDARD 14 )
set( CMAKE_CXX_STANDARD_REQUIRED TRUE )
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++14 -pthread")

project( followme )

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PROJECT_SOURCE_DIR} )


add_executable( demo_pose ${SOURCE_DIR}/main_demo_pose.cpp )



target_link_libraries( demo_pose followmelib)
target_link_libraries( demo_pose ${OpenCV_LIBS} )
target_link_libraries( demo_pose ${realsense2_LIBRARY})
