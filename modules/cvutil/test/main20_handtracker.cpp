/**
 * @file   main20_handtracker.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jan  4 15:55:03 2013
 * 
 * @brief  
 * 
 * 
 */

#define _USE_MATH_DEFINES
#include <math.h>
#include <ctype.h>
#include "cvext_c.h"
// #include "cvmsepf.h"
//#include "cvincrpca.cpp"
#include "cvgabor.h"
#include "cvhandtracker.h"
// #include "cvparticlefilter.h"

typedef struct IcvMouseParam {
  CvPoint center;
  CvRect roi;
  int token;
  IcvMouseParam():token(0){
    center = cvPoint(0,0); memset(&roi,0,sizeof(CvRect));
  }
} IcvMouseParam;
void icvMouseCallback(int evt, int x, int y, int flags, void* param);

int main(int argc, char * argv[])
{

#if 0
  const CvSize imsize = cvSize(320, 240);
#else
  const CvSize imsize = cvSize(160, 120);
#endif
  
  int framecounter=0;
  static int delay = 0; 
  cvNamedWindow("Test");
  
  CvGenericTracker tracker;
  CvHandTracker handtracker(tracker);
  CvCapture * capture = NULL;
  IplImage * rawImage =
      cvCreateImage(imsize, IPL_DEPTH_8U, 3);

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

  // IcvMouseParam mouseparam;
  // cvSetMouseCallback("Test", icvMouseCallback, &mouseparam);

  float timerq_data[33]={20.f,20.f,};
  CvMat timerq = cvMat(33,1,CV_32F,timerq_data);
  int timerqcc=0;
  //--------------------------------------------------
  // START FRAME QUERY
  //--------------------------------------------------
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

CV_TIMER_START();
    if (tracker.initialized()){
      tracker.update(rawImage);
      tracker.m_framecounter=framecounter;
      handtracker.update();
    }else{
      tracker.initialize(rawImage);
    }
    timerq_data[timerqcc++%33]=timer.elapsed()*1000.;
// CV_TIMER_SHOW();

    // if (pfilter.initialized()) // display current frame (optional)
    if (1) // display current frame (optional)
    {
      IplImage * dispImage = cvCloneImage(rawImage);
      // CvRect roi = pfilter.get_window();
      // CvRect roi = tracker.get_window();
      // cvRectangle(dispImage,
      //             cvPoint(roi.x, roi.y),
      //             cvPoint(roi.x+roi.width, roi.y+roi.height),
      //             tracker.valid()?CV_RED:CV_BLUE,
      //             1);
#if 1
      CvBox2D box = handtracker.get_exwin();
      CvRect roi = handtracker.get_window2();// cvBox2DToRect(box);
      if (roi==cvRect(0,0,0,0))
      {
        roi = cvBox2DToRect(box);
      }else
      {
        // CvMat subdisp_stub, * subdisp;
        // subdisp = cvGetSubRect(dispImage, &subdisp_stub, roi);
        // cvSet(subdisp, CV_RED, handtracker.get_segmentation());
        // cvRectangle(dispImage,
        //             cvPoint(roi.x,roi.y),
        //             cvPoint(roi.x+roi.width,roi.y+roi.height),
        //             CV_GREEN, 1);
      }
#if 1
      // cvBoxRectangle(dispImage, box,
      //                tracker.valid()?CV_RED:CV_BLUE,
      //                1);
      box.size.height+=10; box.size.width+=10;
      cvBoxRectangle(dispImage, box,
                     // handtracker.valid()?CV_RED:
                     CV_BLUE,
                     1);
#else
      cvRectangle(dispImage,
                  cvPoint(roi.x,roi.y),
                  cvPoint(roi.x+roi.width,roi.y+roi.height),
                  CV_RED, 1);
      roi.x-=5;roi.y-=5;roi.width+=10;roi.height+=10;
      cvRectangle(dispImage,
                  cvPoint(roi.x,roi.y),
                  cvPoint(roi.x+roi.width,roi.y+roi.height),
                  CV_RED, 1);
#endif
#endif
      char strfps[128];
      sprintf(strfps, "%.0fms", cvAvg(&timerq).val[0]);
      cvDrawLabel(dispImage, strfps, CV_RED, 0);

      cvDrawFrameCount(dispImage, framecounter);
      cvShowImage("Test", dispImage); // CV_WAIT2(10);
      //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_GRAY);
      //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_HSV);
      cvReleaseImageEx(dispImage);
    }

    // KEY PRESS PROCESSING
    int key = cvWaitKey(delay)&0xff;
    if (key==27){
      break;
    }else if (key==' '){
      if (delay){ delay = 0; }else{ delay = 10; }
    }else if (key=='f'){ // skip to next frame
    }else if (key!=0xff){
      fprintf(stderr, "Warning: Unknown key press : 0x%X\n", key);
    } // end of key press processing

  } // end of while loop

  if (!rawImage){
    cvReleaseImage(&rawImage);
    rawImage = NULL;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Test");
}

void icvMouseCallback(int evt, int x, int y, int flags, void* param)
{
  if (((CV_EVENT_LBUTTONDOWN&evt) >> (CV_EVENT_LBUTTONDOWN-1))==1){  // point
	//if (CV_EVENT_FLAG_LBUTTON&flags){ // drag
    fprintf(stderr, "LT(%d,%d)\n", x, y);
    //cvCircle((IplImage*)((IcvMouseParam*)param)->frame, cvPoint(x,y),1,CV_BLACK,-1);
    ((IcvMouseParam*)param)->center = cvPoint(x,y);
    ((IcvMouseParam*)param)->roi = cvRect(x-12,y-12,25,25);
    ((IcvMouseParam*)param)->token = 1;
  }
}

