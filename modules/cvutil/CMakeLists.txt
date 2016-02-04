project(cvutil)

set(CMAKE_MODULE_PATH ${cvutil_SOURCE_DIR}/cmake)

INCLUDE_DIRECTORIES(./include ../cxcore/include ../cv/include ../highgui/include)

add_library(cvutil SHARED
	src/cvkfd.cpp
	src/cvstageddetecthog.cpp
	src/cvinvcomp.cpp
	src/cvparticlestate.cpp
	src/cvshapeprior.cpp
	src/cvchamfer.cpp
	src/cvparticleutil.cpp
	src/cvhog_data.cpp
	src/cvhandvalidator.cpp
	src/cvext_io.cpp
	src/cvparticleobserve.cpp
	src/cvsparsecoding.cpp
	src/cvlda.cpp
	src/cvstip.cpp
	src/cvshapeprior_data.cpp
	src/cvfacecoder.cpp
	src/cvext_common.cpp
	src/cvparticlebase.cpp
	src/cvhomography.cpp
	src/cvsgkfd.cpp
	src/cvhog.cpp
	src/cvstageddetecthaar.cpp
	src/cvhanddetector.cpp
	src/cvio.cpp
	src/cvpwptracker.cpp
	src/cvparticlefilter.cpp
	src/cvimgwarp.cpp
	src/cvpictstruct.cpp
	src/cvhandtracker.cpp
	src/cvsuperres.cpp
	src/cvshapedesc.cpp
	src/cvmotion.cpp
	src/cvaam.cpp
	src/cvextutils.cpp
	src/cvmaxflow.cpp
	src/cvstageddetecthaar_data.cpp
	src/cvhandvalid_data.cpp
	src/cvlevelset.cpp
	src/cvsvm4hog.cpp
	)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(cvutil pthread rt cxcore cv highgui gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(cvutil pthread rt cxcore cv highgui)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)


