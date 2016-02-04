! include ( ../extern/common.pri ){
 error( "Couldn't find the common.pri file!" )
}

INCLUDEPATH += . . ../include \
               ../extern/cxcore/include \
               ../extern/cv/include \
               ../extern/highgui/include
DEFINES += CVAPI_EXPORTS
win32{
LIBS += -llibcxcore -llibcv -llibhighgui -L../lib
}else{
LIBS += -lcxcore -lcv -lhighgui -L../lib
}
DESTDIR = ../lib
TARGET = compvis
win32:TARGET = $$join(TARGET,,lib,) 

# Input
HEADERS += ../include/cvtimer.h \
           ../include/cvsql.h \
           ../include/cvclassifier4fdesc.h \
           ../include/cvparticleutil.h \
           ../include/cvclassifier4ls.h \
           ../include/cvhanddetector.h \
           ../include/cvfacecoder.h \
           ../include/cvparticlefilter.h \
           ../include/cvpwptracker.h \
           ../include/cvtracker.h \
           ../include/cvhog.h \
           ../include/cvext.h \
           ../include/cvparticlebase.h \
           ../include/cvsvm4hog.h \
           ../include/cvsparsecoding.h \
           ../include/cvaam.h \
           ../include/cvshapedesc.h \
           ../include/cvchamfer.h \
           ../include/cvhandtracker.h \
           ../include/cvext_io.h \
           ../include/cvsgkfd4hog.h \
           ../include/cvsgkfd.h \
           ../include/cvhandvalidator.h \
           ../include/cvpictstruct.h \
           ../include/cvkfd.h \
           ../include/cvparticlestate.h \
           ../include/cvimgwarp.h \
           ../include/cvkfd4hog.h \
           ../include/cvshapeprior.h \
           ../include/compvis.h \
           ../include/cvlevelset.h \
           ../include/cvext_c.h \
           ../include/cvunionfind.h \
           ../include/cvlda4hog.h \
           ../include/cvlda.h \
           ../include/cvclassifier.h \
           ../include/cvhomography.h \
           ../include/cvstip.h \
           ../include/cvext.hpp \
           ../include/cvstageddetecthaar.h \
           ../include/cvinvcomp.h \
           ../include/cvparticleobserve.h \
           ../include/cvstageddetecthog.h \
           ../include/cvmaxflow.h
SOURCES += ../src/cvkfd.cpp \
           ../src/cvstageddetecthog.cpp \
           ../src/cvinvcomp.cpp \
           ../src/cvparticlestate.cpp \
           ../src/cvshapeprior.cpp \
           ../src/cvchamfer.cpp \
           ../src/cvparticleutil.cpp \
           ../src/cvhog_data.cpp \
           ../src/cvhandvalidator.cpp \
           ../src/cvext_io.cpp \
           ../src/cvparticleobserve.cpp \
           ../src/cvsparsecoding.cpp \
           ../src/cvlda.cpp \
           ../src/cvstip.cpp \
           ../src/cvshapeprior_data.cpp \
           ../src/cvfacecoder.cpp \
           ../src/cvext_common.cpp \
           ../src/cvparticlebase.cpp \
           ../src/cvhomography.cpp \
           ../src/cvsgkfd.cpp \
           ../src/cvhog.cpp \
           ../src/cvstageddetecthaar.cpp \
           ../src/cvstageddetecthaar_data.cpp \
           ../src/cvhanddetector.cpp \
           ../src/cvio.cpp \
           ../src/cvpwptracker.cpp \
           ../src/cvparticlefilter.cpp \
           ../src/cvimgwarp.cpp \
           ../src/cvpictstruct.cpp \
           ../src/cvhandtracker.cpp \
           ../src/cvsuperres.cpp \
           ../src/cvshapedesc.cpp \
           ../src/cvmotion.cpp \
           ../src/cvaam.cpp \
           ../src/cvextutils.cpp \
           ../src/cvmaxflow.cpp \
           ../src/cvhandvalid_data.cpp \
           ../src/cvlevelset.cpp \
           ../src/cvsvm4hog.cpp

