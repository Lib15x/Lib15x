cmake_minimum_required(VERSION 2.6)

SET(CMAKE_C_COMPILER /usr/bin/gcc-4.9)
SET(CMAKE_CXX_COMPILER /usr/bin/g++-4.9)

PROJECT(CPPLearn)
list (APPEND CMAKE_MODULE_PATH "${CPPLearn_SOURCE_DIR}/../cmake")

#include eigen
SET( EIGEN_DIR "$ENV{EIGEN_DIR}" )
IF( NOT EIGEN_DIR )
  MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN_DIR to the directory of your EIGEN installation.  That would be the parent directory of the Eigen directory.")
ENDIF()
MESSAGE( STATUS "EIGEN_DIR:         " ${EIGEN_DIR} )
INCLUDE_DIRECTORIES(${EIGEN_DIR})

#include google test
SET( GTEST_ROOT "$ENV{GTEST_DIR}" )
IF( NOT GTEST_ROOT )
  MESSAGE( FATAL_ERROR "Please point the environment variable GTEST_DIR to the directory of your google test installation.")
ENDIF()
MESSAGE( STATUS "GTEST_DIR:           " ${GTEST_ROOT} )
FIND_PACKAGE(GTest REQUIRED)
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})
list(APPEND LIBS ${GTEST_BOTH_LIBRARIES})
find_package(BLAS REQUIRED)
list(APPEND LIBS ${BLAS_LIBRARIES})
#MESSAGE( ${LIBS})

#include optimization library ipopt
SET( IPOPT_DIR "$ENV{IPOPT_DIR}" )
IF( NOT IPOPT_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable IPOPT_DIR to the directory of your Ipopt installation.")
ENDIF()
MESSAGE( STATUS "IPOPT_DIR:       " ${IPOPT_DIR} )
INCLUDE_DIRECTORIES(${IPOPT_DIR}/include/coin)
LINK_DIRECTORIES(${IPOPT_DIR}/lib)
list(APPEND LIBS "-lipopt")



#set compilation flag
set(CMAKE_CXX_FLAGS "-Wall -g -Wextra -O3 -DEIGEN_DEBUG -std=c++1y -pthread ${CMAKE_CXX_FLAGS}")

include (CTest)
set (CTEST_PROJECT_NAME "CPPLearn")
enable_testing()

include_directories(${CPPLearn_SOURCE_DIR}/include/)

macro (add_executable_by_name name)
  ADD_EXECUTABLE(${name} ${name}.cc)
  TARGET_LINK_LIBRARIES(${name} ${LIBS})
endmacro (add_executable_by_name)

add_subdirectory(tests)
add_subdirectory(examples)