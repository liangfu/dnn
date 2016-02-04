# set(the_description "Video Analysis")
# ocv_define_module(video opencv_imgproc)
project(video)

set(CMAKE_MODULE_PATH ${video_SOURCE_DIR}/cmake)

INCLUDE_DIRECTORIES(./include ../cxcore/include ../cv/include ../highgui/include)

add_library(video SHARED
	src/bgfg_gmg.cpp
	src/simpleflow.hpp
	src/tvl1flow.cpp
	src/bgfg_gaussmix.cpp
	src/kalman.cpp
	src/precomp.hpp
	src/bgfg_gaussmix2.cpp
	src/simpleflow.cpp
	src/camshift.cpp
	src/lkpyramid.cpp
	src/motempl.cpp
	src/lkpyramid.hpp
	src/video_init.cpp
	src/optflowgf.cpp
	)

add_executable(test_video 
	test/test_simpleflow.cpp
	test/test_estimaterigid.cpp
	test/test_backgroundsubtractor_gbh.cpp
	test/test_camshift.cpp
	test/test_optflowpyrlk.cpp
	test/test_motiontemplates.cpp
	test/test_kalman.cpp
	test/test_tvl1optflow.cpp
	test/test_main.cpp
	test/test_accum.cpp
	)
target_link_libraries(test_video cxcore ts cv video)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(video pthread rt cxcore cv highgui gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(video pthread rt cxcore cv highgui)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)


