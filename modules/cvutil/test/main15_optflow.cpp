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
#include "cvhaarcascadedetector.h"
#include "cvextparticle.h"

static const CvScalar colors[] = 
{
	{{0,0,255}},
	{{0,128,255}},
	{{0,255,255}},
	{{0,255,0}},
	{{255,128,0}},
	{{255,255,0}},
	{{255,0,0}},
	{{255,0,255}}
};

typedef struct stMouseParam{CvPoint2D32f pt;IplImage * img;
stMouseParam(){pt=cvPoint2D32f(0,0);}
} stMouseParam;
void cbMouse(int evt, int x, int y, int flags, void* param);


int main(int argc, char * argv[])
{
  static int framecounter=0;
  const CvSize imsize = cvSize(320,240);
  cvNamedWindow("Test");

  CvParticleState state;
  CvParticleObserve observe; 
  observe.cvParticleObserveInitialize("../data/pcaval.xml","../data/pcavec.xml","../data/pcaavg.xml");
  CvHaarCascadeDetector detector; detector.load();				 
  //CvAbstractTracker tracker;									 
  CvHandTracker tracker;

  CvCapture * capture = NULL;
  if (argc==1) {
    capture = cvCreateCameraCapture(0);
    // set resolution to 320x240
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, imsize.width);
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, imsize.height);
  }else{
    capture = cvCreateFileCapture(argv[1]);
  }
  if (!capture) {fprintf(stderr, "Error: fail to open source video!\n");}

  static CvRect ROIs[50];
  CvParticle *particle = cvCreateParticle( 5/*num_states*/, 100/*num_particles*/, true/*logprob*/ );
  bool pf_initialized=false;

  static int delay = 0; framecounter=350;
  cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
  stMouseParam mouseParam;
  cvSetMouseCallback("Test",cbMouse, &mouseParam);
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
    }else{
      tracker.initialize(rawImage);
      tracker.m_framecounter=framecounter;
    }
  

    // START PROCESSING HERE
    {
	  // draw face rectangles
		mouseParam.img=rawImage;
		float points_data[2];
		CvMat points = cvMat(1,1,CV_32FC2,points_data);
		points.data.fl[0]=mouseParam.pt.x;
		points.data.fl[1]=mouseParam.pt.y;
		cvOpticalFlowPointTrack(tracker.m_currImage, tracker.m_nextImage, &points,cvSize(10,10),2);

      {
        IplImage * dispImage = cvCloneImage(rawImage);
		cvCircle(dispImage,cvPoint(points.data.i[0],points.data.i[1]),2,CV_RED,-1);
        // draw frame count
        cvDrawFrameCount(dispImage, framecounter);
        // show
        cvShowImage("Test", dispImage); CV_WAIT2(10);
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
  cvDestroyWindow("Test");
}

void cbMouse(int evt, int x, int y, int flags, void* param)
{
	if (CV_EVENT_LBUTTONDOWN&evt){  // point
	//if (CV_EVENT_FLAG_LBUTTON&flags){ // drag
		fprintf(stderr, "(%d,%d)\n", x, y);
		cvCircle((IplImage*)((stMouseParam*)param)->img, cvPoint(x,y),1,CV_BLACK,-1);
		((stMouseParam*)param)->pt = cvPoint2D32f(x,y);

	}
}