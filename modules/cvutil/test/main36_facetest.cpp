/**
 * @file   main36_facetest.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Jul 15 10:50:22 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvstageddetecthaar.h"
// #include "cvfacedetector.h"
#include "cvtimer.h"

int main(int argc, char * argv[])
{
  CvSize imsize = cvSize(320,240);
  int i,imgiter;
  CvStagedDetectorHaar detector;
  CvRect rois[1000];

  CvCapture * capture = cvCreateCameraCapture(0);
  const char * imagelist[] = {
    // "../dataset/yalefaces/yalefaces/subject01.normal.png",
    "../data/face/images/judybats.png",
    "../data/face/images/bttf301.png",
    "../data/gesturePalm_JC3train-204.png",
    "../data/gesturePalm_Jiang-0000.png",
    "../data/gesturePalm_Liulu-0300.png",
    "../data/gesturePalm_Steven-0000.png",
    "../data/gesturePalm_JC2-0000.png",
    "../data/face/images/speed.png",
    "../data/face/images/tori-crucify.png",
    "../data/face/images/audrey1.png",
    "../data/face/images/audrey2.png",
    "../data/face/images/bf5275a.png",
    "../data/face/images/bm5205a.png",
    "../data/face/images/bm6290a.png",
    "../data/takeo.pgm",
    "",
  };

  // CvCapture * capture = cvCreateFileCapture(argv[1]);
  // cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  320);
  // cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
  for (imgiter=0;;imgiter++){
#define USE_CAMERA 0
#if USE_CAMERA
    IplImage * raw = cvQueryFrame(capture);
    CvMat * bgr = cvCreateMat(imsize.height,imsize.width,CV_8UC3);
    cvResize(raw,bgr);
#else
    if (strlen(imagelist[imgiter])==0){break;}
IplImage * raw = cvLoadImage(imagelist[imgiter],1);
// IplImage * raw = cvLoadImage("../data/lena.jpg",1);
CvMat bgr_stub,*bgr=cvGetMat(raw,&bgr_stub);
#endif
    CvMat * img = cvCreateMat(bgr->height,bgr->width,CV_8U);
    cvCvtColor(bgr,img,CV_BGR2GRAY);
CV_TIMER_START();
    int nfaces = detector.detect(img,rois);
CV_TIMER_SHOW();
    for (i=0;i<nfaces;i++){
      cvRectangle(bgr,cvPoint(rois[i].x,rois[i].y),
                  cvPoint(rois[i].x+rois[i].width,rois[i].y+rois[i].height),
                  cvScalarAll(255));
    }
    char label[100];sprintf(label,"nfaces: %d",nfaces);
    cvDrawLabel(bgr,label,cvScalarAll(128));

#if USE_CAMERA
    cvShowImage("Test",bgr); CV_WAIT2(10);
    cvReleaseMat(&bgr);
#else
    cvShowImage("Test",bgr); CV_WAIT();
    cvReleaseImage(&raw);
#endif
    cvReleaseMat(&img);
  }

  cvReleaseCapture(&capture);
  
  return 0;
}
