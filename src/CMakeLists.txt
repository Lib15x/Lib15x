cmake_minimum_required(VERSION 2.6)

PROJECT(CPPLearn)

list (APPEND CMAKE_MODULE_PATH "${CPPLearn_SOURCE_DIR}/../cmake")

#include eigen
SET( EIGEN_DIR "$ENV{EIGEN_DIR}" )
IF( NOT EIGEN_DIR )
  MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN_DIR to the directory of your EIGEN installation.  That would be the parent directory of the Eigen directory.")
ENDIF()
MESSAGE( STATUS "EIGEN_DIR:         " ${EIGEN_DIR} )
INCLUDE_DIRECTORIES(${EIGEN_DIR})

set(CMAKE_CXX_FLAGS "-Wall -g -Wextra -O3 -DEIGEN_DEBUG -std=c++1y ${CMAKE_CXX_FLAGS}")

include (CTest)
set (CTEST_PROJECT_NAME "CPPLearn")
enable_testing()

include_directories(${CPPLearn_SOURCE_DIR})

macro (add_executable_by_name name)
  ADD_EXECUTABLE(${name} ${name}.cc)
  TARGET_LINK_LIBRARIES(${name})
endmacro (add_executable_by_name)

add_subdirectory(tests)
add_subdirectory(examples)