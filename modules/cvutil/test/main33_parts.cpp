/**
 * @file   main32_parts.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jun 28 18:11:32 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvpictstruct.h"
#include "cvstageddetecthaar.h"
#include "cvtimer.h"

int main(int argc, char * argv[])
{
  CvStagedDetectorHaar detector;
  CvPartsStructure parts;
  // psmodel.train((char*)"../dataset/parts/upperbody.txt");
  parts.train((char*)"../dataset/parts/parts.txt");

  CvSize imsize = cvSize(320,240);
  int i,delay=10;
  // CvFaceDetector detector;
  CvRect rois[1000];

#if 0
  CvCapture * capture = cvCreateCameraCapture(0);
#else
  CvCapture * capture =
      cvCreateFileCapture("../data/gesturePalm_JC3train.avi");
#endif
  // cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  320);
  // cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
  cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 120);
  for (;;){
    IplImage * raw = cvQueryFrame(capture);
    CvMat * bgr = cvCreateMat(imsize.height,imsize.width,CV_8UC3);
    cvResize(raw,bgr);
    CvMat * img = cvCreateMat(bgr->rows,bgr->cols,CV_8U);
    cvCvtColor(bgr,img,CV_BGR2GRAY);

#if 0
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
#endif

    if (!parts.initialized()){
      parts.initialize(img,cvBox2D(117, 51, 24, 30, -10));
    }
    
    cvShowImage("Test",bgr);
    int key=cvWaitKey(delay)&0xff;
    if (key==27){
      cvReleaseMat(&bgr);
      cvReleaseMat(&img);break;
    }else if(key==' '){delay=(delay==0)?10:0;}

    cvReleaseMat(&bgr);
    cvReleaseMat(&img);
  }

  cvReleaseCapture(&capture);
  
  return 0;
}
