cmake_minimum_required(VERSION 2.8.2)

project(gpu)

INCLUDE_DIRECTORIES(./include ../cxcore/include ../ts/include)

set(Srcs 
  src/basic.cpp
  src/cmdoptions.cpp
  src/cmdparser.cpp
  src/gemm.cpp
  src/oclobject.cpp
# src/utils.cpp
# src/yuv_utils.cpp
)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )
find_package( OpenCL REQUIRED )

add_library(gpu SHARED ${Srcs})

# if (APPLE)
#   target_compile_definitions(gpu INTERFACE "-framework OpenCL")
# endif()

# add_executable(test_gpu
# 	test/test_main.cpp
# 	)
# target_link_libraries(test_gpu cxcore ts gpu)

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} ${OpenCL_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${OpenCL_CXX_FLAGS}" )
	IF(WIN32)
		TARGET_LINK_LIBRARIES(gpu cxcore OpenCL)
	ELSEIF(APPLE)
		TARGET_LINK_LIBRARIES(gpu pthread cxcore iomp5)
    SET_TARGET_PROPERTIES(gpu PROPERTIES LINK_FLAGS "-framework OpenCL")
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(gpu pthread rt cxcore gomp OpenCL)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSEIF(APPLE)
		TARGET_LINK_LIBRARIES(gpu pthread cxcore OpenCL)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(gpu pthread rt cxcore OpenCL)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

