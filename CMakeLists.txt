cmake_minimum_required(VERSION 2.8.2)

if(POLICY CMP0020)
  cmake_policy(SET CMP0020 NEW)
endif()

if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW) # CMake 3.0
endif()

project(compvis)

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

add_definitions(-DCVAPI_EXPORTS)

add_subdirectory(modules/zlib)
add_subdirectory(modules/znzlib)

add_subdirectory(modules/cxcore)
add_subdirectory(modules/cv)
# add_subdirectory(modules/video)
add_subdirectory(modules/highgui)
add_subdirectory(modules/ts)
add_subdirectory(modules/ml)
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
set( Srcs 
	# res/appicon.rc
	# src/bneditablelabel.cpp
	# src/bnmainwindow.cpp
	# src/bnmainwindow_dti.cpp
	# src/bnmainwindow_gui.cpp
	# src/bnmainwindow_odf.cpp
	# src/bnoverlaymgr.cpp
	# src/bnsliceimageview.cpp
	# src/bnpropertywidget.cpp 
	# src/bnvtkcanvasmgr.cpp 
	# src/bnutility.cpp 
	# src/bndlgcreateroi_impl.cpp
	# src/bndlgfiberpruning_impl.cpp
	# src/bndlgregistration_impl.cpp
	src/main.cpp
	src/mainwindow.cpp
	# src/mediaplayer.cpp
	src/videoplayer.cpp
	src/videowidget.cpp
	src/objectdialog.cpp
	# src/videowidgetsurface.cpp
)

set( Hdrs 
	# include/bneditablelabel.h
	# include/bnmainwindow.h
	# include/bndlgregistration_impl.h
	# include/bndlgcreateroi_impl.h
	# include/bndlgfiberpruning_impl.h
	# include/bnoverlaymgr.h
	# include/bnsliceimageview.h
	# include/_vatool.h
	# include/bnutility.h
	# include/bnpropertywidget.h 
	# include/bnvtkcanvasmgr.h
	# include/mediaplayer.h
	include/mainwindow.h
	include/videoplayer.h
	include/videowidget.h
	include/objectdialog.h
	# include/videowidgetsurface.h
)

set( UIs 
	src/mainwindow.ui
	# src/settings.ui
	# src/bndlgregistration.ui 
	# src/bndlgcreateroi.ui 
	# src/bndlgfiberpruning.ui
)
set( QRC_Files res/mediaplayer.qrc)

include_directories(
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
	include
	src
	modules/znzlib
	modules/cxcore/include
	modules/cv/include
	modules/cvutil/include
	modules/highgui/include
	modules/ml/include
	modules/video/include
	modules/latentssvm-v0.12
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
#   add_executable(compvis ${Srcs} ${Hdrs_MOC} ${UI_Srcs} ${RCC_Files})
#   qt5_use_modules(compvis Core Gui Widgets)
#   target_link_libraries(compvis 
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
#   # add_executable(compvis ${Srcs} ${Hdrs_MOC} ${UI_Srcs} ${RCC_Files})
# 	# # qt4_use_modules(compvis Core Gui Network)
#   # target_link_libraries(compvis 
# 	# 	# opencv_core opencv_highgui opencv_imgproc
# 	# 	cxcore cv highgui 
# 	# 	# ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY} phonon -L/usr/local/Trolltech/Qt-4.8.6/lib
# 	# 	${QT_LIBRARIES} 
# 	# 	)
# endif()


# add_executable(test027_pwptracker src/test027_pwptracker.cpp)
# target_link_libraries(test027_pwptracker cxcore cvutil cv video highgui )

# add_executable(test042_optflow src/test042_optflow.cpp)
# target_link_libraries(test042_optflow cxcore cvutil cv video highgui )

# add_executable(test043_optflow src/test043_optflow.cpp)
# target_link_libraries(test043_optflow cxcore cvutil cv video highgui )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
    # target_link_libraries(test044_cnn cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES}) # video
		set( OPENMP_LIBRARIES "")
	ELSE(WIN32)
	  # target_link_libraries(test044_cnn cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} gomp) # video
		set( OPENMP_LIBRARIES gomp)
	endif(WIN32)
else( OPENMP_FOUND )
	# target_link_libraries(test044_cnn cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES}) # video 
  set( OPENMP_LIBRARIES "")
endif( OPENMP_FOUND )

# add_executable(test044_cnn src/test044_cnn.cpp src/ConvNN.cpp)
# target_link_libraries(test044_cnn cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

# add_executable(test046_dram src/test046_dram.cpp src/dram.cpp)
# target_link_libraries(test046_dram cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES}) # video

add_executable(network src/network_main.cpp src/network.cpp)
target_link_libraries(network cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

add_executable(transfer_mnist_lenet src/transfer_mnist_lenet.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_lenet cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

add_executable(transfer_mnist_twodigit src/transfer_mnist_twodigit.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_twodigit cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

add_executable(transfer_mnist_addition src/transfer_mnist_addition.cpp  src/network.cpp)
target_link_libraries(transfer_mnist_addition cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

add_executable(transfer_addition src/transfer_addition.cpp  src/network.cpp)
target_link_libraries(transfer_addition cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})

add_executable(transfer_svhn_dram src/transfer_svhn_dram.cpp  src/network.cpp)
target_link_libraries(transfer_svhn_dram cxcore cvutil cv highgui ml ${FFMPEG_LIBRARIES} ${OPENMP_LIBRARIES})



