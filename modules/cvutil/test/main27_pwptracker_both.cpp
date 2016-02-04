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
// #include "cvext_c.h"
// #include "cvmsepf.h"
//#include "cvincrpca.cpp"
// #include "cvgabor.h"
#include "cvhandtracker.h"
// #include "cvpfilter.h"

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
  fprintf(stderr, "64bit: sizeof(void*): %d\n", sizeof(void*));
  fprintf(stderr, "endian: 256 in memory is presented as 0x%x\n", 256);
#if 1
  const CvSize imsize = cvSize(320, 240);
#else
  const CvSize imsize = cvSize(160, 120);
#endif
  
  int framecounter=0;

  static int delay = 10; 

  cvNamedWindow("Test");
  
  CvGenericTracker tracker;
  CvHandTracker handtracker(tracker);
  CvCapture * capture = NULL;
  IplImage * grayImage = NULL;
  IplImage * rawImage =
      cvCreateImage(imsize, IPL_DEPTH_8U, 3);

  //--------------------------------------------------
  // SET CAMERAS
  //--------------------------------------------------
  if ( (argc==1) ||                                             // no arg
       ((argc>1)&&isdigit(argv[1][0])&&(strlen(argv[1])==1)) )  // 1st arg
  {
    capture = cvCaptureFromCAM((argc==1)?0:atoi(argv[1]));//cvCreateCameraCapture((argc==1)?0:atoi(argv[1]));
    // set resolution to `imsize`
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  imsize.width);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imsize.height);
  }else{
    capture =      (argc>1)?cvCaptureFromAVI(argv[1]):0;//capture =      (argc>1)?cvCreateFileCapture(argv[1]):0;
    framecounter = (argc>2)?atoi(argv[2])               :20;
    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
  }
  //capture = cvCaptureFromCAM(-1);
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  int TZG = 1;//0;//
  //if(argc > 2)
  //{
	 // TZG = atoi(argv[2]);
  //}

  IcvMouseParam mouseparam;
  cvSetMouseCallback("Test", icvMouseCallback, &mouseparam);

  float timerq_data[30]={20.f,20.f,};
  CvMat timerq = cvMat(30,1,CV_32F,timerq_data);
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

	//tzg
	//cvFlip(rawImage, rawImage);

    if (grayImage==NULL) {
      grayImage =
          cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
    }
    cvCvtColor(rawImage, grayImage, CV_BGR2GRAY);

CV_TIMER_START();
    if (tracker.initialized()){
      tracker.update(rawImage);
      tracker.m_framecounter=framecounter;
	  if(!TZG)
	  {
		handtracker.update();
	  }
	  else
	  {
		  tracker.CalcSilhouette();//tzg
		  handtracker.myUpdate();//tzg
	  }
    }else{
      tracker.initialize(rawImage);
    }
    timerq_data[timerqcc++%30]=timer.elapsed()*1000.;

    // ---------------------------------
    // display current frame (optional)
    if (!TZG)
    {
      // IplImage * dispImage = cvCloneImage(rawImage);
      IplImage * dispImage = cvCloneImage(tracker.m_currColorImage);

      CvBox2D inbox = handtracker.get_inwin();
      CvBox2D exbox = handtracker.get_exwin();
      CvRect roi = handtracker.get_window2();// cvBox2DToRect(box);

      // ---------------------------------
      // show object contour
      if (handtracker.initialized() && handtracker.bw_full)
      {
        if (1) // leave boundary pixels only
        {
          uchar * ptr =handtracker.bw_full->data.ptr;
          int j,k,step=handtracker.bw_full->step/sizeof(uchar);
          for (j=0;j<imsize.height-1;j++,ptr+=step)
            for (k=0;k<imsize.width-1;k++)
            {
              ptr[k]=((ptr[k+1]!=ptr[k]) || ((ptr+step)[k]!=ptr[k]))?255:0;
            }    
        }
        cvSet(dispImage, CV_RED, handtracker.bw_full);
      }

      // ---------------------------------
      // show template location
      if (1)
      {
         //box.size.height+=10; box.size.width+=10;
        //cvBoxRectangle(dispImage, inbox,
        //               // handtracker.valid()?CV_RED:
        //               CV_BLUE, 2);
        //cvBoxRectangle(dispImage, exbox,
        //               // handtracker.valid()?CV_RED:
        //               CV_BLUE, 1);

		  if(handtracker.phi)
		  {
			  CvPoint pts[4];
			  int nc = handtracker.phi->width;
			  int nr = handtracker.phi->height;
			  float* pWarp = (float*)handtracker.warp_p->data.fl;
			  
			  float pTrans[6];
			  pTrans[0] = pWarp[0];
			  pTrans[1] = -pWarp[1];
			  pTrans[2] = pWarp[2];
			  pTrans[3] = pWarp[1];
			  pTrans[4] = pWarp[0];
			  pTrans[5] = pWarp[3];

			  int xx = 0;
			  int yy = 0;
			  pts[0].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[0].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = nc;
			  yy = 0;
			  pts[1].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[1].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = nc;
			  yy = nr;
			  pts[2].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[2].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = 0;
			  yy = nr;
			  pts[3].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[3].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];		 

			  cvLine(dispImage, pts[0], pts[1], CV_BLUE, 1);
			  cvLine(dispImage, pts[2], pts[1], CV_BLUE, 1);
			  cvLine(dispImage, pts[2], pts[3], CV_BLUE, 1);
			  cvLine(dispImage, pts[0], pts[3], CV_BLUE, 1);

			  int border = 8;
			  xx = border;
			  yy = border;
			  pts[0].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[0].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = nc - border;
			  yy = 0 + border;
			  pts[1].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[1].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = nc - border;
			  yy = nr - border;
			  pts[2].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[2].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];
			  xx = 0 + border;
			  yy = nr - border;
			  pts[3].x = pTrans[0]*xx + pTrans[1]*yy + pTrans[2];
			  pts[3].y = pTrans[3]*xx + pTrans[4]*yy + pTrans[5];		 

			  cvLine(dispImage, pts[0], pts[1], CV_BLUE, 2);
			  cvLine(dispImage, pts[2], pts[1], CV_BLUE, 2);
			  cvLine(dispImage, pts[2], pts[3], CV_BLUE, 2);
			  cvLine(dispImage, pts[0], pts[3], CV_BLUE, 2);
		  }
      }

      // ---------------------------------
      // show labels
      char strfps[128];
      // sprintf(strfps, "%.0fms - %d",
      //         cvAvg(&timerq).val[0], handtracker.status());
      sprintf(strfps, "%.0fms - %s",
              cvAvg(&timerq).val[0],
              (handtracker.status()==0)?"open":
              (handtracker.status()==1)?"close":"lost");
      cvDrawLabel(dispImage, strfps, CV_RED, 0,2);
      cvDrawFrameCount(dispImage, framecounter);
      cvShowImage("Test", dispImage); //CV_WAIT2(10);

      // KEY PRESS PROCESSING
      int key = cvWaitKey(delay)&0xff;
      if (key==27){
        break;
      }else if (key==' '){
        if (delay){ delay = 0; }else{ delay = 5; }
      }else if (key=='f'){ // skip to next frame
      }else if (key!=0xff){
        fprintf(stderr, "Warning: Unknown key press : 0x%X\n", key);
      } // end of key press processing

      cvReleaseImageEx(dispImage);
    }

	//tzg
	if(TZG)
	{
 		IplImage * dispImage = cvCloneImage(tracker.m_currColorImage);

		char strfps[128];
		//sprintf(strfps, "%.0fms-%d",cvAvg(&timerq).val[0], framecounter);//, handtracker.status());

		sprintf(strfps, "%.0fms - %s",
			cvAvg(&timerq).val[0],
			(handtracker.m_status==0)?"open":
			(handtracker.m_status==1)?"close":"lost");

		cvDrawLabel(dispImage, strfps, CV_RED, 0);

		handtracker.myDisplay(dispImage);

		cvShowImage("Test", dispImage); //CV_WAIT2(10);

		// KEY PRESS PROCESSING
		delay = 1;
		int key = cvWaitKey(delay)&0xff;
		if (key==27){
			break;
		}else if (key==' '){
			if (delay){ delay = 0; }else{ delay = 1; }
		}else if (key=='f'){ // skip to next frame
		}else if (key!=0xff){
			fprintf(stderr, "Warning: Unknown key press : 0x%X\n", key);
		} // end of key press process

		cvReleaseImageEx(dispImage);
	}

  } // end of while loop

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

