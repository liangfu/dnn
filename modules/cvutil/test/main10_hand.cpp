/**
 * @file   main10_hand.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Dec 10 13:35:53 2012
 * 
 * @brief  
 * 
 * 
 */


#include "cvext.h"
#include "cvext_hand.h"

int main(int argc, char * argv[])
{
  static int framecounter=0;
  
  //CvAbstractTracker tracker;
  CvHandTracker tracker;

  CvCapture * capture = NULL;
  if (argc==1) {
    capture = cvCreateCameraCapture(0);
    // set resolution to 320x240
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
  }else{
    capture = cvCreateFileCapture(argv[1]);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  static int delay = 0; 
  while(1)
  {
    if (0){
      cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter++);
    }else{
      framecounter++;
    }
    IplImage * rawImage = cvQueryFrame(capture);
    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}
    if (tracker.initialized()){
      tracker.update(rawImage);
      tracker.m_framecounter=framecounter;
    }else{
      tracker.initialize(rawImage);
    }

    // START PROCESSING HERE
    {
      {
        //IplImage * dispImage = cvCloneImage(rawImage);
		IplImage * dispImage = cvCloneImage(tracker.m_diffImage);
        cvDrawFrameCount(dispImage, framecounter);
        cvShowImage("Test", dispImage); CV_WAIT2(30);
        cvReleaseImageEx(dispImage);
      }
    }
    
    int key = cvWaitKey(delay)&0xff;
    if (key==27){
      break;
    }else if (key==' '){
      if (delay){ delay = 0; }else{ delay = 30; }
    }else if (key=='f'){ // skip to next frame
    }else if (key!=0xff){
      fprintf(stderr, "Warning: Unknown key press : %c\n", key);
    } // end of key press processing
  } // end of video

  cvReleaseCapture(&capture);
}
