cmake_minimum_required(VERSION 2.8.7)

project(latentssvm)

INCLUDE_DIRECTORIES(. ./svm_light)

add_library(latentssvm SHARED
svm_light/svm_common.c
svm_light/svm_hideo.c
 # svm_light/svm_classify.c
 svm_light/svm_learn.c
 # svm_light/svm_learn_main.c
svm_struct_latent_api.c
 # svm_struct_latent_cccp.c
 #  svm_struct_latent_classify.c
	)

add_executable(svm_latent_learn svm_struct_latent_cccp.c )
add_executable(svm_latent_classify svm_struct_latent_classify.c )

target_link_libraries(svm_latent_learn latentssvm m )
target_link_libraries(svm_latent_classify latentssvm m )

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(latentssvm pthread rt gomp)
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(latentssvm pthread rt)
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)

