cmake_minimum_required(VERSION 2.8.2)

project(ts)

INCLUDE_DIRECTORIES(./include ../cxcore/include)

add_library(ts SHARED
	src/ts_gtest.cpp
	src/ts_perf.cpp
	src/ts_func.cpp
	src/ts.cpp
	src/ts_arrtest.cpp
	src/gpu_test.cpp
	src/gpu_perf.cpp
	)

target_link_libraries(ts cxcore highgui)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(ts pthread rt cxcore gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(ts pthread rt cxcore)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

