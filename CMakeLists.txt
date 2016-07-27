cmake_minimum_required(VERSION 2.8.2)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW) # CMake 3.0
endif()

project(DNNLAB)

# this is EXTREMELY important to put these setting before add_library/executable
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 

SET(CMAKE_INSTALL_RPATH "$ORIGIN/../lib:$ORIGIN/")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

# add a custom "postfix" to static and shared libraries when in Debug build
SET(CMAKE_DEBUG_POSTFIX "_debug")

IF(APPLE)
#   SET(CMAKE_XCODE_ATTRIBUTE_GCC_VERSION "com.apple.compilers.llvmgcc42")
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
endif()

add_definitions(-DCVAPI_EXPORTS)

add_subdirectory(thirdparty/zlib)
add_subdirectory(thirdparty/znzlib)

add_subdirectory(modules/cxcore)
add_subdirectory(modules/cv)
# add_subdirectory(modules/video)
add_subdirectory(modules/highgui)
add_subdirectory(modules/ts)
add_subdirectory(modules/dnn)
add_subdirectory(modules/cvutil)
# add_subdirectory(modules/pose)
# add_subdirectory(modules/latentssvm-v0.12)

# find_package(OpenCV REQUIRED opencv_core opencv_highgui)
find_package( OpenMP )
if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
ENDIF(OPENMP_FOUND)

# Set your files and resources here

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
	include
	src
	thirdparty/znzlib
	modules/cxcore/include
	modules/cv/include
	modules/cvutil/include
	modules/highgui/include
	modules/dnn/include
)

# Instruct CMake to run moc automatically when needed.
# set(CMAKE_AUTOMOC ON)

# if(VTK_QT_VERSION VERSION_GREATER "4")
#   find_package(Qt5Widgets REQUIRED QUIET)
#   set(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
#   set(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
#   set(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
#   qt5_wrap_ui(UI_Srcs ${UIs})
#   qt5_wrap_cpp(Hdrs_MOC ${Hdrs})
#   qt5_add_resources(RCC_Files ${QRC_Files})
#   # CMAKE_AUTOMOC in ON so the MocHdrs will be automatically wrapped.
#   # add_executable(QtVTKRenderWindows ${Srcs} ${Hdrs} ${UI_Srcs} ${MOC_Hdrs})
#   # qt5_use_modules(QtVTKRenderWindows Core Gui Widgets)
#   # target_link_libraries(QtVTKRenderWindows vtkDICOM vis utils ${VTK_LIBRARIES})
#   add_executable(dnn ${Srcs} ${Hdrs_MOC} ${UI_Srcs} ${RCC_Files})
#   qt5_use_modules(dnn Core Gui Widgets)
#   target_link_libraries(dnn 
# 		cxcore
# 		# ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} phonon Qt4::QtCore
# 		${QT_LIBRARIES} 
# 		)
# else()
#   find_package(Qt4 4.5.3 REQUIRED QtCore QtGui)
# 	if (MSVC)
# 	  set(QT_ROOT_DIR "D:/Qt/4.8.6")
# 	else(MSVC)
# 		set(QT_ROOT_DIR "/usr/local/Trolltech/Qt-4.8.6")
# 	endif (MSVC)
#   include(${QT_USE_FILE})
#   # Use what VTK built with
#   # set(QT_QMAKE_EXECUTABLE ${VTK_QT_QMAKE_EXECUTABLE} CACHE FILEPATH "")
#   # set(QT_MOC_EXECUTABLE ${VTK_QT_MOC_EXECUTABLE} CACHE FILEPATH "")
#   # set(QT_UIC_EXECUTABLE ${VTK_QT_UIC_EXECUTABLE} CACHE FILEPATH "")
#   qt4_wrap_cpp(Hdrs_MOC ${Hdrs})
#   qt4_wrap_ui(UI_Srcs ${UIs})
#   qt4_add_resources(RCC_Files ${QRC_Files})
#   # add_executable(dnn ${Srcs} ${Hdrs_MOC} ${UI_Srcs} ${RCC_Files})
# 	# # qt4_use_modules(dnn Core Gui Network)
#   # target_link_libraries(dnn 
# 	# 	# opencv_core opencv_highgui opencv_imgproc
# 	# 	cxcore cv highgui 
# 	# 	# ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} phonon -L/usr/local/Trolltech/Qt-4.8.6/lib
# 	# 	${QT_LIBRARIES} 
# 	# 	)
# endif()

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
    # target_link_libraries(test044_cnn cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES}) # video
		set( OPENMP_LIBRARIES "")
  ELSEIF(APPLE)
	  SET( OPENMP_LIBRARIES iomp5)
	ELSE(WIN32)
	  # target_link_libraries(test044_cnn cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} gomp) # video
		set( OPENMP_LIBRARIES gomp)
	endif(WIN32)
else( OPENMP_FOUND )
	# target_link_libraries(test044_cnn cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES}) # video 
  set( OPENMP_LIBRARIES "")
endif( OPENMP_FOUND )

add_executable(annotator src/annotator_main.cpp)
target_link_libraries(annotator cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(annotator PROPERTIES SUFFIX "_debug")
endif()

add_executable(review src/review_main.cpp)
target_link_libraries(review cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(review PROPERTIES SUFFIX "_debug")
endif()

add_executable(transform src/transform_main.cpp)
target_link_libraries(transform cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transform PROPERTIES SUFFIX "_debug")
endif()

# add_executable(categorize src/categorize_main.cpp src/network.cpp)
# target_link_libraries(categorize cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
# if (CMAKE_BUILD_TYPE STREQUAL Debug)
# 	set_target_properties(categorize PROPERTIES SUFFIX "_debug")
# endif()

add_executable(network src/network_main.cpp src/network.cpp)
target_link_libraries(network cxcore cvutil cv highgui dnn ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(network PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_mnist_lenet src/transfer_mnist_lenet.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_lenet cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_mnist_lenet PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_mnist_twodigit src/transfer_mnist_twodigit.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_twodigit cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_mnist_twodigit PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_mnist_multi src/transfer_mnist_multi.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_multi cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_mnist_multi PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_mnist_localize src/transfer_mnist_localize.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_localize cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_mnist_localize PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_mnist_addition src/transfer_mnist_addition.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_addition cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_mnist_addition PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_addition src/transfer_addition.cpp  src/network.cpp)
target_link_libraries(transfer_addition cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_addition PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_svhn_dram src/transfer_svhn_dram.cpp  src/network.cpp)
target_link_libraries(transfer_svhn_dram cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_svhn_dram PROPERTIES SUFFIX "_debug")
endif()

add_executable(transfer_primate_convnet src/transfer_primate_convnet.cpp  src/network.cpp)
target_link_libraries(transfer_primate_convnet cxcore cvutil cv highgui dnn 
  ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})
if (CMAKE_BUILD_TYPE STREQUAL Debug)
	set_target_properties(transfer_primate_convnet PROPERTIES SUFFIX "_debug")
endif()



