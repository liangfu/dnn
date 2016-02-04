#define _USE_MATH_DEFINES
#include <math.h>
#include "cvext.h"
#include "cvmsepf.h"
#include "cvincrpca.cpp"
#include "cvgabor.h"

typedef struct IcvMouseParam {
  CvPoint center;
  CvRect roi;
  int token;
  IcvMouseParam():token(0){
    center = cvPoint(0,0); memset(&roi,0,sizeof(int)*4);
  }
} IcvMouseParam;
void icvMouseCallback(int evt, int x, int y, int flags, void* param);

int main(int argc, char * argv[])
{
  int framecounter=0;
  static int delay = 0; 
  cvNamedWindow("Test");
  
  CvMSEPF pfilter;
  // CvAbstractTracker tracker;
  CvGeneticTracker tracker;
  CvCapture * capture = NULL;
  IplImage * grayImage = NULL;
  IplImage * rawImage =
      cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);

  //--------------------------------------------------
  // SET CAMERAS
  //--------------------------------------------------
  if ( (argc==1) ||                                             // no arg
       ((argc>1)&&isdigit(argv[1][0])&&(strlen(argv[1])==1)) )  // 1st arg
  {
    capture = cvCreateCameraCapture((argc==1)?0:atoi(argv[1]));
    // set resolution to 320x240
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  320);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 240);
  }else{
    capture =      (argc>1)?cvCreateFileCapture(argv[1]):0;
    framecounter = (argc>2)?atoi(argv[2])               :100;
    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  IcvMouseParam mouseparam;
  cvSetMouseCallback("Test",icvMouseCallback, &mouseparam);

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
    if ( (rawImage_data->height==rawImage->height)&&
         (rawImage_data->width ==rawImage->width) )
    {
      cvCopy(rawImage_data, rawImage);
    }else{
      cvResize(rawImage_data, rawImage);
    }

    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}
    if (grayImage==NULL) {
      grayImage =
          cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
    }
    cvCvtColor(rawImage, grayImage, CV_BGR2GRAY);

CV_TIMER_START();
    if (tracker.initialized()){
      tracker.update(rawImage);
      tracker.m_framecounter=framecounter;
    }else{
      tracker.initialize(rawImage);
    }

    if (1) // START PROCESSING HERE
    {
      double dShiftSize = /* (argc>3)?atof(argv[3]): */3.0;
      double dWinSize   = /* (argc>4)?atof(argv[4]): */0.2;
      double dRadius    = /* (argc>5)?atof(argv[5]): */0.3;

      pfilter.config(dShiftSize, dShiftSize,
                     dWinSize, dWinSize,
                     dRadius);

      if (mouseparam.token) // mouse press captured
      {
        cvShowImage("Test", rawImage); CV_WAIT();
        pfilter.initialize(mouseparam.roi);

        if (0) // save pointed rectangle (optional)
        {
          IplImage * saveImage =
              cvCreateImage(cvSize(mouseparam.roi.width,
                                   mouseparam.roi.height),8,3);
          cvCropImageROI(rawImage, saveImage,
                         cvRect32fFromRect(mouseparam.roi));
          cvSaveImage("../data/palm000.png", saveImage);
          cvReleaseImage(&saveImage);
        }
        mouseparam.token=0;
      }

      if (pfilter.pf_initialized)
      {
        pfilter.observe(rawImage);
      }
CV_TIMER_SHOW();

      if (1) // display current frame (optional)
      {
        IplImage * dispImage = cvCloneImage(rawImage);

        cvDrawFrameCount(dispImage, framecounter);
        cvShowImage("Test", dispImage);
        //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_GRAY);
        //cvShowImageEx("Test", tracker.m_mhiImage, CV_CM_HSV);
        cvReleaseImageEx(dispImage);
      }
    } // END OF FRAME PROCESSING

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
  } // end of video

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

