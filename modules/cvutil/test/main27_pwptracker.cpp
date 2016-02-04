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
void print_help(int argc, char * argv[]);

int main(int argc, char * argv[])
{
  // fprintf(stderr, "sizeof(void*)==%d?\n", sizeof(void*));
  // fprintf(stderr, "4660(0x1234)==0x%x?\n", 4660);
#if 1
  CvSize imsize = cvSize(320, 240);
  int initframe = 0;
  for (int i=2;i<argc;i++){
    if (!strcmp("--start-frame",argv[i]) ||
        !strcmp("-s",argv[i])) {
      if (argc<=i){
        fprintf(stderr, "ERROR: too few arguments!\n");
        print_help(argc,argv);
        return 1;
      }
      initframe=atoi(argv[++i]);
    }else if (!strcmp("--geometry",argv[i])||!strcmp("-g",argv[i])){
      if (argc==i){
        fprintf(stderr, "ERROR: too few arguments!\n");
        print_help(argc,argv);
        return 1;
      }
      if (strcmp("320x240",argv[i+1])==0){
        imsize=cvSize(320, 240);
      }else if (strcmp("160x120",argv[i+1])==0){
        imsize=cvSize(160,120);
      }else{
        fprintf(stderr, "ERROR: unsupported geometry %s!\n",argv[i+1]);
        print_help(argc,argv);
        return 1;
      }
      i++;
    }else{
      fprintf(stderr, "ERROR: unknown argument %s!\n",argv[i]);
      print_help(argc,argv);
      return 1;
    }
  }
#else
  const CvSize imsize = cvSize(160, 120);
#endif
  
  int framecounter=0;
  static int delay = 0;
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
    capture = cvCreateCameraCapture((argc==1)?0:atoi(argv[1]));
    // set resolution to `imsize`
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  imsize.width);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imsize.height);
  }else{
    capture =      (argc>1)?cvCreateFileCapture(argv[1]):0;
    framecounter = initframe;//(argc>2)?atoi(argv[2]):20;
    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  // IcvMouseParam mouseparam;
  // cvSetMouseCallback("Test", icvMouseCallback, &mouseparam);

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
    if (!rawImage_data) {
      fprintf(stderr, "Info: end of video!\n"); break;
    }

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
    if (tracker.initialized()){
      tracker.update(rawImage);
// CV_TIMER_SHOW();
      tracker.m_framecounter=framecounter;
      handtracker.update();
    }else{
      tracker.initialize(rawImage);
    }
    timerq_data[timerqcc++%30]=timer.elapsed()*1000.;

    // ---------------------------------
    // display current frame
    {
      // IplImage * dispImage = cvCloneImage(rawImage);
      IplImage * dispImage = cvCloneImage(tracker.m_currColorImage);

      // CvBox2D inbox = handtracker.get_inwin();
      CvBox2D inbox = handtracker.get_pfwin();
      CvBox2D exbox = handtracker.get_exwin();
      //CvRect   roi  = handtracker.get_window2();// cvBox2DToRect(box);

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
        // box.size.height+=10; box.size.width+=10;
        cvBoxRectangle(dispImage, inbox,
                       // handtracker.valid()?CV_RED:
                       CV_RED, 1);
        cvBoxRectangle(dispImage, exbox,
                       // handtracker.valid()?CV_RED:
                       CV_BLUE, 1);
      }

      // ---------------------------------
      // show labels
      char strfps[128];
      // sprintf(strfps, "%.0fms - %d",
      //         cvAvg(&timerq).val[0], handtracker.status());
      sprintf(strfps, "%s - %.0fms",
              (handtracker.status()==0)?"open":
              (handtracker.status()==1)?"close":"lost",
              cvAvg(&timerq).val[0]);
      cvDrawLabel(dispImage, strfps, CV_RED, 0,2);
      cvDrawFrameCount(dispImage, framecounter);
      cvShowImage("Test", dispImage); //CV_WAIT2(10);

      // KEY PRESS PROCESSING
      int key = cvWaitKey(delay)&0xff;
      if (key==27){
        break;
        cvReleaseImageEx(dispImage);
      }else if (key==' '){
        if (delay){ delay = 0; }else{ delay = 5; }
      }else if (key=='f'){ // skip to next frame
      }else if (key=='o'){
        char saveImageStr[128]; char tmpstr[64];
        strncpy(tmpstr,argv[1]+8,strlen(argv[1]+8));
        tmpstr[strlen(argv[1]+8)-4]='\0';
        sprintf(saveImageStr,
                "../dataset/palm/pos/%s-%d.png",
                tmpstr,framecounter);
        fprintf(stderr, "saving to %s\n", saveImageStr); 
        cvSaveImage(saveImageStr, tracker.m_currColorImage);
        FILE * fp = fopen("../dataset/palm/open.txt", "a+");
        fprintf(fp,"%s %.1f %.1f %.1f %.1f\n",
                saveImageStr,
                inbox.center.x-inbox.size.width*.5,
                inbox.center.y-inbox.size.height*.5,
                inbox.size.width,inbox.size.height);
        fclose(fp);
      }else if (key=='c'){
        char saveImageStr[128]; char tmpstr[64];
        strncpy(tmpstr,argv[1]+8,strlen(argv[1]+8));
        tmpstr[strlen(argv[1]+8)-4]='\0';
        sprintf(saveImageStr,
                "../dataset/palm/pos/%s-%d.png",
                tmpstr,framecounter);
        fprintf(stderr, "saving to %s\n", saveImageStr); 
        cvSaveImage(saveImageStr, tracker.m_currColorImage);
        FILE * fp = fopen("../dataset/palm/close.txt", "a+");
        fprintf(fp,"%s %.1f %.1f %.1f %.1f\n",
                saveImageStr,
                inbox.center.x-inbox.size.width*.5,
                inbox.center.y-inbox.size.height*.5,
                inbox.size.width,inbox.size.height);
        fclose(fp);
      }else if (key!=0xff){
        fprintf(stderr, "Warning: Unknown key press : 0x%X\n", key);
      } // end of key press processing

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
  // cvDestroyWindow("Test");
  cvDestroyAllWindows();

  return 0;
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

void print_help(int argc,char * argv[])
{
  fprintf(stderr,
          "Example:\n"
          "   %s video.avi --start-frame 20 --geometry 320x240\n",
          argv[0]);
}
