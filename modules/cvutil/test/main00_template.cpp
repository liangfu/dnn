/**
 * @file   main26_optflow.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Wed Feb  6 11:57:25 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvinvcomp.h"
#include "cvext.h"
#include <ctype.h>

int main(int argc, char * argv[])
{

#if 1
  const CvSize imsize = cvSize(320, 240);
#else
  const CvSize imsize = cvSize(160, 120);
#endif
  
  int framecounter=0;
  static int delay = 0; 
  cvNamedWindow("Test");
  
  // CvPFilter pfilter;
  // CvMSEPF pfilter;
  // CvAbstractTracker tracker;
  // CvGeneticTracker tracker;
  // CvHandTracker tracker;
  CvCapture * capture = NULL;
  IplImage * grayImage = NULL;
  IplImage * rawImage = cvCreateImage(imsize, IPL_DEPTH_8U, 3);

  //--------------------------------------------------
  // SET CAMERAS
  //--------------------------------------------------
  if ( (argc==1) ||                                             // no arg
       ((argc>1)&&isdigit(argv[1][0])&&(strlen(argv[1])==1)) )  // 1st arg
  {
    capture = cvCreateCameraCapture((argc==1)?0:atoi(argv[1]));
    // set resolution to `imsize`
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  imsize.width);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imsize.height);
  }else{
    capture =      (argc>1)?cvCreateFileCapture(argv[1]):0;
    framecounter = (argc>2)?atoi(argv[2])               :100;
    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  float timerq_data[33]={20.f,20.f,};
  CvMat timerq = cvMat(33,1,CV_32F,timerq_data);
  int timerqcc=0;

  while(1)
  {
    if (0){
      cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter++);
	}else{
      framecounter++;
    }

    IplImage * rawImage_data = cvQueryFrame(capture);
    if (!rawImage_data) {fprintf(stderr, "Info: end of video!\n"); break;}

    if ( (rawImage_data->height==rawImage->height)&&
         (rawImage_data->width ==rawImage->width) )
    {
      cvCopy(rawImage_data, rawImage);
    }else{
      cvResize(rawImage_data, rawImage);
    }

    if (grayImage==NULL) {
      grayImage =
          cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
    }
    cvCvtColor(rawImage, grayImage, CV_BGR2GRAY);

CV_TIMER_START();
    {
      
    }
    timerq_data[timerqcc++%33]=timer.elapsed()*1000.;
  
    if (1) // display current frame (optional)
    {
      IplImage * dispImage = cvCloneImage(rawImage);
      char strfps[128];
      sprintf(strfps, "%.0fms", cvAvg(&timerq).val[0]);
      cvDrawLabel(dispImage, strfps, CV_RED, 0);

      cvDrawFrameCount(dispImage, framecounter);
      cvShowImage("Test", dispImage); CV_WAIT2(10);
      //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_GRAY);
      //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_HSV);
      cvReleaseImageEx(dispImage);
    }

    // KEY PRESS PROCESSING
    int key = cvWaitKey(delay)&0xff;
    if (key==27){
      break;
    }else if (key==' '){
      if (delay){ delay = 0; }else{ delay = 30; }
    }else if (key=='f'){ // skip to next frame
    }else if (key!=0xff){
      fprintf(stderr, "Warning: Unknown key press : 0x%X\n", key);
    } // end of key press processing
  }

  if (grayImage!=NULL){
    cvReleaseImage(&grayImage);
    grayImage = NULL;
  }
  if (!rawImage){
    cvReleaseImage(&rawImage);
    rawImage = NULL;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Test");

  return 0;
}
