cmake_minimum_required(VERSION 2.8.2)

project(dnn)

INCLUDE_DIRECTORIES(./include ../cxcore/include 
  ../cvutil/include ../cv/include ../highgui/include ../ts/include)

add_library(dnn SHARED
	src/dnn.cpp
	src/merge_layer.cpp
	src/conv_layer.cpp
	src/fc_layer.cpp
	src/input_layer.cpp
	src/repeat_layer.cpp
	src/pool_layer.cpp
	src/relu_layer.cpp
	src/rnn_layer.cpp
	src/sigmoid_layer.cpp
	src/softmax_layer.cpp
	src/imgwarp_layer.cpp
	src/tdist_layer.cpp
	src/utils.cpp
	)

add_executable(test_dnn
	test/test_dnn.cpp
	test/test_main.cpp
	)
target_link_libraries(test_dnn cxcore ts dnn)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
		TARGET_LINK_LIBRARIES(dnn cxcore cvutil cv highgui gpu)
	ELSEIF(APPLE)
		TARGET_LINK_LIBRARIES(dnn pthread cxcore cvutil cv highgui gpu iomp5)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(dnn pthread rt cxcore cvutil cv highgui gpu gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
		TARGET_LINK_LIBRARIES(dnn cxcore cvutil cv highgui gpu)
	ELSEIF(APPLE)
		TARGET_LINK_LIBRARIES(dnn pthread cxcore cvutil cv highgui gpu)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(dnn pthread rt cxcore cvutil cv highgui gpu)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

