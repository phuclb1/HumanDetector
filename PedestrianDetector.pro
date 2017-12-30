TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH+= /usr/local/include/opencv2
LIBS += -L/usr/local/lib
LIBS += -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_bgsegm -lopencv_objdetect -lpthread -ldlib -lcblas -llapack


SOURCES += main.cpp \
    Tracker.cpp \
    InitRectDrawer.cpp \
    VideoReader.cpp
HEADERS += \
    Tracker.h \
    InitRectDrawer.h \
    VideoReader.h


