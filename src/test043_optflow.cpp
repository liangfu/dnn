// Pyramid L-K optical flow example
//
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

#include "cvext.h"
#include "cvtracker.h"

const int MAX_CORNERS = 2500;

int main(int argc, char** argv) 
{
  // GLOBAL SETTINGS
  static int framecounter=0;
  const CvSize imsize = cvSize(320,240);
  int delay = 0;
  
  const int win_size = 10;
  CvSize pyr_sz = cvSize( imsize.width+8, imsize.height/3 );
  IplImage * pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
  IplImage * pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
  IplImage * rawImage_resized = cvCreateImage( imsize, IPL_DEPTH_8U, 3);

  cvNamedWindow("Test");
  CvGenericTracker tracker;

  // LOAD INPUT FILE
  CvCapture * capture = NULL;
  if (argc==1) {
    capture = cvCreateCameraCapture(0);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, imsize.width);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imsize.height);
  }else{
    capture = cvCreateFileCapture(argv[1]);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");return 0;}
  cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);

  // START ENDLESS LOOP
  while(1)
  {
	// GET NEXT FRAME
    if (1){
      cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter++);
    }else{
      framecounter++;
    }
    IplImage * rawImage = cvQueryFrame(capture);
	cvResize(rawImage,rawImage_resized);
    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}
    if (tracker.initialized()){
      tracker.update(rawImage_resized);
    }else{
      tracker.initialize(rawImage_resized);
      tracker.m_framecounter=framecounter;
    }

    // START PROCESSING HERE
    {
	  // Initialize, load two images from the file system, and
	  // allocate the images and other structures we will need for
	  // results.
	  CvMat * imgA = tracker.m_currImage;
	  IplImage * imgB = tracker.m_nextImage;
	  IplImage * imgC = cvCloneImage(rawImage_resized);
  
	  // The first thing we need to do is get the features
	  // we want to track.
	  IplImage * eig_image = cvCreateImage( imsize, IPL_DEPTH_32F, 1 );
	  IplImage * tmp_image = cvCreateImage( imsize, IPL_DEPTH_32F, 1 );
	  int corner_count = MAX_CORNERS;
	  CvPoint2D32f* cornersA = new CvPoint2D32f[ MAX_CORNERS ];
	  cvGoodFeaturesToTrack(imgA,eig_image,tmp_image,cornersA,&corner_count,0.01,5.0,0,3,0,0.04);
	  cvFindCornerSubPix(imgA,cornersA,corner_count,cvSize(win_size,win_size),cvSize(-1,-1),
						 cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03));

	  // Call the Lucas Kanade algorithm
	  char features_found[ MAX_CORNERS ];
	  float feature_errors[ MAX_CORNERS ];
	  CvPoint2D32f * cornersB = new CvPoint2D32f[ MAX_CORNERS ];
	  cvCalcOpticalFlowPyrLK(imgA,imgB,pyrA,pyrB,
							 cornersA,cornersB,corner_count,cvSize( win_size,win_size ),
							 5,features_found,feature_errors,
							 cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ),
							 (framecounter<2)?0:CV_LKFLOW_PYR_B_READY);

	  // Now make some image of what we are looking at:
	  for( int i=0; i<corner_count; i++ ) {
		if( features_found[i]==0|| feature_errors[i]>550 ) {
		  fprintf(stderr,"error=%f\n",feature_errors[i]);continue;
		}
		CvPoint p0 = cvPoint(cvRound( cornersA[i].x ),cvRound( cornersA[i].y ));
		CvPoint p1 = cvPoint(cvRound( cornersB[i].x ),cvRound( cornersB[i].y ));
		cvLine( imgC, p0, p1, CV_RGB(255,0,0), 1 );
	  }

	  cvShowImage("Test",imgC);
	  cvReleaseImage(&imgC);
	  cvReleaseImage(&eig_image);
	  cvReleaseImage(&tmp_image);
	  delete [] cornersA;
	  delete [] cornersB;
	}
	
	// DISPLAY PROCESSING RESULT
	int key = cvWaitKey(delay)&0xff;
	if (key==27){
	  break;
	}else if (key==' '){
	  if (delay){ delay = 0; }else{ delay = 30; }
	}else if (key=='f'){ // skip to next frame
	}else if (key=='S'){ // skip to next frame
	  framecounter+=10;fprintf(stderr,"framecount:%d\n",framecounter);
	}else if (key=='Q'){ // skip to next frame
	  framecounter=MAX(1,framecounter-10);fprintf(stderr,"framecount:%d\n",framecounter);
	}else if (key!=0xff){
	  fprintf(stderr, "Warning: Unknown key press : %c\n", key);
	} // end of key press processing
  } // end of video

  cvReleaseImage(&pyrA);
  cvReleaseImage(&pyrB);
  cvReleaseImage(&rawImage_resized);

  return 0;
}

