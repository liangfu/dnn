cmake_minimum_required(VERSION 2.8.2)

project(ml)

INCLUDE_DIRECTORIES(./include ../cxcore/include ../cvutil/include ../cv/include ../highgui/include)

add_library(ml SHARED
	src/em.cpp
	src/testset.cpp
	src/gbt.cpp
	src/ann_mlp.cpp
	src/precomp.hpp
	src/data.cpp
	src/ertrees.cpp
	src/nbayes.cpp
	src/estimate.cpp
	src/knearest.cpp
	src/rtrees.cpp
	src/ml_init.cpp
	src/boost.cpp
	src/cnn.cpp
	src/inner_functions.cpp
	src/svm.cpp
	src/tree.cpp
	)

# add_executable(test_ml
# 	test/test_mltests.cpp
# 	test/test_emknearestkmeans.cpp
# 	test/test_mltests2.cpp
# 	test/test_gbttest.cpp
# 	test/test_save_load.cpp
# 	test/test_main.cpp
# 	)
# target_link_libraries(test_ml cxcore ts ml)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(ml pthread rt cxcore cvutil cv highgui gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(ml pthread rt cxcore cvutil cv highgui)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

