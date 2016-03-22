cmake_minimum_required(VERSION 2.8.2)

project(zlib)

INCLUDE_DIRECTORIES(./include)

add_library(zlib
  src/crc32.h    
  src/gzguts.h   
  src/infback.c
  src/inflate.c
  src/trees.c
  src/zutil.h
  src/adler32.c
  src/deflate.c
  src/gzlib.c
  src/inffast.c
  src/inflate.h
  src/trees.h
  src/compress.c
  src/deflate.h
  src/gzread.c
  src/inffast.h
  src/inftrees.c
  src/uncompr.c
  src/crc32.c
  src/gzclose.c
  src/gzwrite.c
  src/inffixed.h
  src/inftrees.h
  src/zutil.c
	)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(zlib pthread rt gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(zlib pthread rt)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

