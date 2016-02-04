/**
 * @file   main21_videocap.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jan  4 18:07:38 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"

void print_help(int argc, char * argv[]);

int main(int argc, char* argv[])
{
  print_help(argc, argv);
  int framecounter = 0;
  int delay = 0;
  CvCapture * capture;
  CvSize imsize = cvSize(320, 240);
  IplImage * rawImage =
      cvCreateImage(imsize, IPL_DEPTH_8U, 3);
  cvNamedWindow("Test");

  // SET CAMERAS
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

  CvVideoWriter * writer =
      cvCreateVideoWriter(argc>1?argv[1]:"video.avi",
                          CV_FOURCC('P','I','M','1'),
                          // CV_FOURCC('M','J','P','G'),
                          30, imsize, 1 /*iscolor*/);
  
  CV_WAIT();
  
  // START 
  while (1)
  {
    IplImage * rawImage_data = cvQueryFrame(capture);
    if ( (rawImage_data->height==rawImage->height)&&
         (rawImage_data->width ==rawImage->width) )
    {
      cvCopy(rawImage_data, rawImage);
    }else{
      cvResize(rawImage_data, rawImage);
    }
    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}

    if (1)                                      // start processing
    {
      cvWriteFrame(writer, rawImage);
      
      if (1) // display current frame (optional)
      {
        IplImage * dispImage = cvCloneImage(rawImage);

        cvDrawFrameCount(dispImage, framecounter);
        cvShowImage("Test", dispImage);
        cvReleaseImage(&dispImage);
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
  }

  cvReleaseVideoWriter(&writer);
  cvReleaseImage(&rawImage);
  cvReleaseCapture(&capture);
  cvDestroyAllWindows();

  return 0;
}

void print_help(int argc, char * argv[])
{
  fprintf(stderr, "Usage:\n"
          "\t%s $video.avi\n", argv[0]);
}
