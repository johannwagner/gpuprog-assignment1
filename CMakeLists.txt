# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project( assignment1)

set( CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( SRC_DIR ${ROOT_DIR}/src)

set(SOURCE
   ${SRC_DIR}/main.cpp
)

if(APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -std=c++14 -Wno-unused-variable  -Wno-unused-parameter")
endif(APPLE)

if(UNIX AND NOT APPLE)
  set( CMAKE_CXX_FLAGS "-W -Wall -std=c++14 -Wno-unused-variable  -Wno-unused-parameter -lpthread")
endif(UNIX AND NOT APPLE)


add_executable( assignment1 ${SOURCE})
