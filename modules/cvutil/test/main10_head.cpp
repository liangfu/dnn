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

int main(int argc, char * argv[])
{
  static int framecounter=0;
  const CvSize imsize = cvSize(320,240);

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
  while(1)
  {
    if (1){
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
		int nfaces = 0; //if (framecounter%300!=0) {nfaces=detector.detect(rawImage, ROIs);}
	  if (!pf_initialized){
		  if (0!=detector.detect(rawImage, ROIs)){
			  CvParticleState::cvParticleStateConfig( particle, cvSize(320,240), cvParticleState (
				  //std_x,std_y,std_w,std_h,std_r
				  3.10,3.0,2.10,2.10,1.1) );

			  // initialize particle filter
			  CvParticle *init_particle = cvCreateParticle( 5/*num_states*/, 1 );
			  CvRect32f region32f = cvRect32fFromRect( cvRect(ROIs[0].x-2+ROIs[0].width/2, ROIs[0].y-2+ROIs[0].height/2, 5, 5) );
			  CvBox32f box = cvBox32fFromRect32f( region32f ); // centerize
			  CvRect32f s = cvParticleState( box.cx, box.cy, box.width, box.height, 0.0 );
			  cvParticleStateSet( init_particle, 0, s );
			  cvParticleInit( particle, init_particle );
			  cvReleaseParticle( &init_particle );
			  pf_initialized=1;
		  }
	  }else if (nfaces){
		  // initialize particle filter
		  CvParticle *init_particle = cvCreateParticle( 5/*num_states*/, 1 );
		  CvRect32f region32f = cvRect32fFromRect( cvRect(ROIs[0].x-2+ROIs[0].width/2, ROIs[0].y-2+ROIs[0].height/2, 5, 5) );
		  CvBox32f box = cvBox32fFromRect32f( region32f ); // centerize
		  CvRect32f s = cvParticleState( box.cx, box.cy, box.width, box.height, 0.0 );
		  cvParticleStateSet( init_particle, 0, s );
		  cvParticleInit( particle, init_particle );
		  cvReleaseParticle( &init_particle );
		  pf_initialized=1;

		  // Draw new particles
		  cvParticleTransition( particle );
		  // Measurement
		  observe.cvParticleObserveMeasure( particle, rawImage );

		  // Draw all particles
		  for( int i = 0; i < particle->num_particles; i++ )
		  {
			  CvRect32f s = cvParticleStateGet( particle, i );
			  cvParticleStateDisplay( s, rawImage, CV_RGB(0,0,255) );
		  }
		  // Draw most probable particle
		  //printf( "Most probable particle's state\n" );
		  int maxp_id = cvParticleGetMax( particle );
		  CvRect32f maxs = cvParticleStateGet( particle, maxp_id );
		  cvParticleStateDisplay( maxs, rawImage, CV_RGB(255,0,0) );
	  }if (pf_initialized){
		  // Draw new particles
		  cvParticleTransition( particle );
		  // Measurement
		  observe.cvParticleObserveMeasure( particle, rawImage );

		  // Draw all particles
		  for( int i = 0; i < particle->num_particles; i++ )
		  {
			  CvRect32f s = cvParticleStateGet( particle, i );
			  cvParticleStateDisplay( s, rawImage, CV_RGB(0,0,255) );
		  }
		  // Draw most probable particle
		  //printf( "Most probable particle's state\n" );
		  int maxp_id = cvParticleGetMax( particle );
		  CvRect32f maxs = cvParticleStateGet( particle, maxp_id );
		  cvParticleStateDisplay( maxs, rawImage, CV_RGB(255,0,0) );
	  }

      {
        IplImage * dispImage = cvCloneImage(rawImage);

		for (int i = 0; i < nfaces; i++){
			cvRectangle(dispImage, 
				cvPoint(ROIs[i].x, ROIs[i].y), 
				cvPoint(ROIs[i].x+ROIs[i].width, ROIs[i].y+ROIs[i].height), 
				colors[i%8], 1);
		}
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
}