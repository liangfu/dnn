/**
 * @file   main26_optflow.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Wed Feb  6 11:57:25 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvinvcomp_.h"
#include "cvext.h"

void mouse_cb(int evt, int x, int y, int flags, void * center)
{
  // pointing
  if (((CV_EVENT_LBUTTONDOWN&evt) >> (CV_EVENT_LBUTTONDOWN-1))==1)
  {
    *((CvPoint*)center) = cvPoint(x,y);
  }
}

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
  CvAbstractTracker tracker;
  // CvGeneticTracker tracker;
  // CvHandTracker tracker;
  CvCapture * capture = NULL;
  IplImage * grayImage = NULL;
  CvMat * gray32f = cvCreateMat(imsize.height, imsize.width, CV_32F);
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

  CvPoint location=cvPoint(-1,-1);
  CvSize tmplt_size = cvSize(24, 24);
  cvSetMouseCallback("Test", mouse_cb, &location);

  float timerq_data[33]={20.f,20.f,};
  CvMat timerq = cvMat(33,1,CV_32F,timerq_data);
  int timerqcc=0;

  CvInvComp invcomp;
  CvMat * warp_p = cvCreateMat(2, 3, CV_32F); cvZero(warp_p);
  CvMat * tmplt =
      cvCreateMat(tmplt_size.height, tmplt_size.width, CV_32F);

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
    cvConvert(grayImage, gray32f);

CV_TIMER_START();
    if ( (location.x>0) && (location.y>0) && (warp_p->data.fl[2]<1) )
    {
      warp_p->data.fl[0] = 1.;          warp_p->data.fl[1] = 0.;
      warp_p->data.fl[2] = location.x-tmplt_size.width/2.;
      warp_p->data.fl[3] = 0.;          warp_p->data.fl[4] = 1.;
      warp_p->data.fl[5] = location.y-tmplt_size.height/2.;
      fprintf(stderr, "location: %d, %d\n", location.x, location.y);
      // fprintf(stderr, "\nwarp_p:\n");
      // cvPrintf(stderr, "%f ", warp_p);

      invcomp.warp_a(gray32f, tmplt, warp_p);
      // cvGetSubRect(gray32f, tmplt,
      //              cvRect(warp_p->data.fl[2], warp_p->data.fl[5],
      //                     tmplt_size.width, tmplt_size.height));
    }
    if (warp_p->data.fl[2]>0 && warp_p->data.fl[5]>0)
    {
      // invcomp.warp_a(gray32f, tmplt, warp_p);
      // cvWarpAffine(gray32f, tmplt, warp_p);
      // cvGetSubRect(gray32f, tmplt,
      //              cvRect(warp_p->data.fl[2]-tmplt->width/2.,
      //                     warp_p->data.fl[5]-tmplt->height/2.,
      //                     tmplt->width, tmplt->height));
      // cvShowImageEx("Test", gray32f, CV_CM_GRAY); CV_WAIT();
      // cvShowImageEx("Test", tmplt, CV_CM_GRAY); CV_WAIT();
      invcomp.affine_ic(gray32f, tmplt, warp_p, 100, 1);
      // if (fabs(sqrt(warp_p->data.fl[0]*warp_p->data.fl[0]+
      //               warp_p->data.fl[1]*warp_p->data.fl[1])-1.)>0.5)
      // {
      //   warp_p->data.fl[0] = 1.;          warp_p->data.fl[1] = 0.;
      //   warp_p->data.fl[3] = 0.;          warp_p->data.fl[4] = 1.;
      // }
      invcomp.warp_a(gray32f, tmplt, warp_p);
      fprintf(stderr, "warp_p:\n");
      cvPrintf(stderr, "%f ", warp_p);
      // fprintf(stderr, "warp: %f, %f\n",
      //         warp_p->data.fl[2], warp_p->data.fl[5]);
    }
    timerq_data[timerqcc++%33]=timer.elapsed()*1000.;
  
    if (1) // display current frame (optional)
    {
      IplImage * dispImage = cvCloneImage(rawImage);
      char strfps[128];
      sprintf(strfps, "%.0fms", cvAvg(&timerq).val[0]);
      cvDrawLabel(dispImage, strfps, CV_RED, 0);

      cvDrawFrameCount(dispImage, framecounter);
      cvCircle(dispImage,
               cvPoint(cvRound(warp_p->data.fl[2]+tmplt_size.width/2.),
                       cvRound(warp_p->data.fl[5]+tmplt_size.height/2.)),
               5, CV_RED);
      cvShowImage("Test", dispImage); CV_WAIT2(10);
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
  cvReleaseMat(&gray32f);
  cvReleaseMat(&warp_p);
  cvReleaseMat(&tmplt);
  cvReleaseCapture(&capture);
  cvDestroyWindow("Test");

  return 0;
}

