cmake_minimum_required(VERSION 2.8.2)

project(cxcore)

INCLUDE_DIRECTORIES(./include)

add_library(cxcore SHARED
	src/algorithm.cpp
	src/alloc.cpp
	src/arithm.cpp
	src/array.cpp
	src/cmdparser.cpp
	src/convert.cpp
	src/copy.cpp
	src/datastructs.cpp
	src/drawing.cpp
	src/dxt.cpp
	src/gl_core_3_1.cpp
	src/gl_core_3_1.hpp
	src/glob.cpp
	src/gpumat.cpp
	src/lapack.cpp
	src/mathfuncs.cpp
	src/matmul.cpp
	src/matop.cpp
	src/matrix.cpp
	src/opengl_interop.cpp
	src/opengl_interop_deprecated.cpp
	src/out.cpp
	src/parallel.cpp
	src/persistence.cpp
	src/precomp.hpp
	src/rand.cpp
	src/stat.cpp
	src/system.cpp
	src/tables.cpp
	)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(cxcore pthread rt gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(cxcore pthread rt)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

