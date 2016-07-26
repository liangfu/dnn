project(highgui)

set(CMAKE_MODULE_PATH ${highgui_SOURCE_DIR}/cmake)

if(BUILD_SHARED_LIBS)
  add_definitions(-DHIGHGUI_EXPORTS)
endif()

# find_package(GStreamer REQUIRED)
find_package(FFMPEG REQUIRED)
find_package(GTK2 REQUIRED)

MESSAGE( STATUS "FFMPEG_INCLUDE_DIR: " ${FFMPEG_INCLUDE_DIR} )
MESSAGE( STATUS "FFMPEG_LIBRARIES: " ${FFMPEG_LIBRARIES} )
MESSAGE( STATUS "GTK2_INCLUDE_DIRS: " ${GTK2_INCLUDE_DIRS} )

INCLUDE_DIRECTORIES(./include ../cxcore/include ../cv/include ../ts/include
	${FFMPEG_INCLUDE_DIR} ${GTK2_INCLUDE_DIRS}) # ${GSTREAMER_INCLUDE_DIRS} 

set(Srcs 
	src/bitstrm.cpp
	src/cap_android.cpp
	src/cap_cmu.cpp
	src/cap.cpp
	# src/cap_dc1394.cpp
	# src/cap_dc1394_v2.cpp
	# src/cap_dshow.cpp
	src/cap_ffmpeg.cpp
	# src/cap_giganetix.cpp
	# src/cap_gstreamer.cpp
	src/cap_images.cpp
	src/cap_intelperc.cpp
	# src/cap_libv4l.cpp
	# src/cap_mil.cpp
	# src/cap_msmf.cpp
	# src/cap_openni.cpp
	# src/cap_pvapi.cpp
	# src/cap_qt.cpp
	# src/cap_tyzx.cpp
	# src/cap_unicap.cpp
	src/cap_v4l.cpp
	# src/cap_vfw.cpp
	# src/cap_ximea.cpp
	# src/cap_xine.cpp
	src/grfmt_base.cpp
	src/grfmt_bmp.cpp
	# src/grfmt_exr.cpp
	src/grfmt_imageio.cpp
	src/grfmt_jpeg2000.cpp
	src/grfmt_jpeg.cpp
	src/grfmt_png.cpp
	src/grfmt_pxm.cpp
	src/grfmt_sunras.cpp
	src/grfmt_tiff.cpp
	src/loadsave.cpp
	src/utils.cpp
	src/window.cpp
	# src/window_carbon.cpp
	src/window_gtk.cpp
	# src/window_QT.cpp
  src/window_w32.cpp
)

if (APPLE)
  set(Srcs ${Srcs} src/window_cocoa.mm)
  FIND_LIBRARY(COCOA_LIBRARY Cocoa)
  FIND_LIBRARY(APPKIT_LIBRARY AppKit)
  MARK_AS_ADVANCED (COCOA_LIBRARY APPKIT_LIBRARY)
  SET(EXTRA_LIBS ${COCOA_LIBRARY} ${APPKIT_LIBRARY})
	message(${EXTRA_LIBS})
  set_source_files_properties(${Srcs} PROPERTIES COMPILE_FLAGS "-x objective-c++")
endif()

add_library(highgui SHARED ${Srcs})

add_executable(test_highgui
	test/test_fourcc.cpp
	test/test_framecount.cpp
	test/test_ffmpeg.cpp
	test/test_gui.cpp
	test/test_positioning.cpp
	test/test_drawing.cpp
	test/test_video_io.cpp
	test/test_basic_props.cpp
	test/test_grfmt.cpp
	test/test_video_pos.cpp
	test/test_main.cpp
	)
target_link_libraries(test_highgui cxcore ts cv highgui)

#---------------------------------------------------------------------
# Find OpenMP
find_package( OpenMP )

if( OPENMP_FOUND )
  set( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
  set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
	IF(WIN32)
		TARGET_LINK_LIBRARIES(highgui cxcore cv ${FFMPEG_LIBRARIES} )
	ELSEIF(APPLE)
		TARGET_LINK_LIBRARIES(highgui pthread cxcore cv ${EXTRA_LIBS} png jpeg tiff jasper iomp5)
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(highgui pthread rt cxcore cv png jpeg tiff jasper v4l1 
			${FFMPEG_LIBRARIES} ${GTK2_LIBRARIES} gomp) # ${GSTREAMER_LIBRARIES} 
	ENDIF(WIN32)
ELSE (OPENMP_FOUND)
	IF(WIN32)
		TARGET_LINK_LIBRARIES(highgui cxcore cv ${FFMPEG_LIBRARIES} )
	ELSE(WIN32)
		TARGET_LINK_LIBRARIES(highgui pthread rt cxcore cv png jpeg tiff jasper v4l1 
			${FFMPEG_LIBRARIES} ${GTK2_LIBRARIES}) #  ${GSTREAMER_LIBRARIES}
	ENDIF(WIN32)
ENDIF(OPENMP_FOUND)


