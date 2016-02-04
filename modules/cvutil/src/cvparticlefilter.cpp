/**
 * @file   cvpfilter.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jan  7 17:50:21 2013
 * 
 * @brief  
 *   
 */

#include "cvparticlefilter.h"
#include "cvimgwarp.h"

CVAPI(void) cvParticleStateConfig( CvParticle * p,
                                   const CvSize imsize, CvBox2D std )
{
  // config dynamics model
  CvMat * dynamicsmat =
      cvCreateMat( p->num_states, p->num_states, CV_64F );
  cvSetIdentity(dynamicsmat, cvRealScalar(1));

  // config random noise standard deviation
  // CvRNG rng = cvRNG( time( NULL ) );
  double stdarr[] = {
    // std.x, std.y,
    // std.width, std.height,
    std.center.x,   std.center.y,
    std.size.width, std.size.height,
    std.angle
  };
  CvMat stdmat = cvMat( p->num_states, 1, CV_64FC1, stdarr );

  // config minimum and maximum values of states
  // lowerbound, upperbound, circular flag (useful for degree)
  // lowerbound == upperbound to express no bounding
  // const CvSize tsize = cvSize(24.*(imsize.width /160.),
  //                             30.*(imsize.height/120.));
  const CvSize tsize = cvSize(30.*(imsize.width /160.),
                              36.*(imsize.height/120.));
  const float space = // space for rotation of box
      (10.*(imsize.width /160.)+                    // additional bound
       MAX(tsize.width, tsize.height))*0.1415+      // rotated bound
      1.0;                                          // additional pixel
  double boundarr[] = {
    space, imsize.width - (tsize.width +1.) - space, false,   // position.x
    space, imsize.height -(tsize.height+1.) - space, false,   // position.y
    12.*(imsize.width/160.),  tsize.width , false, // winsize.x
    15.*(imsize.height/160.), tsize.height, false, // winsize.y
    // 0, 360, true                              // dtheta + circular flag
    -30, 30, false                 // dtheta
  };

  CvMat boundmat = cvMat( p->num_states, 3, CV_64FC1, boundarr );

  //cvParticleSetDynamics( p, dynamicsmat );
  cvConvert(dynamicsmat, p->dynamics);
  // cvParticleSetNoise( p, rng, &stdmat );
  p->rng = cvRNG( time( NULL ) );
  cvConvert(&stdmat,p->std);
  // cvParticleSetBound( p, &boundmat );
  cvConvert(&boundmat,p->bound);
  
  cvReleaseMat(&dynamicsmat);
}

void CvParticleFilter::initialize(CvRect roi)
{
  // initialize particle filter  
  CvParticle * init_particle = cvCreateParticle( N_p/*num_states*/, 1 );
  float lin = m_imsize.width/160.f;
  float bound = 5.*lin;
  float m_iLinMultiplier=m_imgY->width/160.;
  fprintf(stderr, "roi: %d,%d,%d,%d\n", roi.x,roi.y,roi.width,roi.height);
  if ( (roi.x<1) || (roi.y<1) ||
       (roi.width >m_imsize.width-1) ||
       (roi.height>m_imsize.height-1) )
  {
    fprintf(stderr, "WARNING: initializing at image boundary, rejected!\n");
    return;
  }
  CvRect initROI = cvRect(roi.x+bound,
                          roi.y+bound,
                          roi.width -bound*2.,
                          roi.height-bound*2.);
  CvBox2D box = cvBox2DFromRect( initROI ); // centerize

  cvParticleStateSet( init_particle, 0, box );
  cvParticleInit( m_particle, init_particle );
  cvReleaseParticle( &init_particle );

  {
    float dx=1.8,dy=1.8,dw=.01,dh=.01,dtheta=.05;
    CvBox2D box = cvBox2D(dx*lin, dy*lin, dw*lin, dh*lin, dtheta);
    cvParticleStateConfig( m_particle, m_imsize, box);
  }

  {
    // CvMat * imgYpatch  =
    //     cvCreateMat(initROI.height, initROI.width, CV_8U);
    // CvMat * imgUpatch  =
    //     cvCreateMat(initROI.height, initROI.width, CV_8U);
    // CvMat * imgVpatch  =
    //     cvCreateMat(initROI.height, initROI.width, CV_8U);
    // cvGetSubRect(m_imgY, imgYpatch, initROI);
    // cvGetSubRect(m_imgU, imgUpatch, initROI);
    // cvGetSubRect(m_imgV, imgVpatch, initROI);
    // // CV_SHOW(imgYpatch);
    // m_observer.update(imgYpatch, imgUpatch, imgVpatch);

    // int pnr=(roi.height/2)*2,pnc=(roi.width/2)*2;
    // CvMat * imgYpatch = cvCreateMat(pnr, pnc, CV_8U);
    // CvMat * imgUpatch = cvCreateMat(pnr, pnc, CV_8U);
    // CvMat * imgVpatch = cvCreateMat(pnr, pnc, CV_8U);
    // CvMat * mask = cvCreateMat(pnr, pnc, CV_8U);
    // float warp_p_data[3]={1,roi.x,roi.y};
    // CvMat warp_p = cvMat(3,1,CV_32F,warp_p_data);
    // icvWarp(m_imgY, imgYpatch, &warp_p);
    // icvWarp(m_imgU, imgUpatch, &warp_p);
    // icvWarp(m_imgV, imgVpatch, &warp_p);

    // CvMat * subdx = cvCreateMat(pnr, pnc, CV_8U);
    // CvMat * subdy = cvCreateMat(pnr, pnc, CV_8U);
    // icvWarp(m_tracker.dx, subdx, &warp_p);
    // icvWarp(m_tracker.dy, subdy, &warp_p);

    // m_observer.update(imgYpatch, imgUpatch, imgVpatch, mask);
    m_observer.initialize(roi);

    // cvReleaseMat(&subdx);
    // cvReleaseMat(&subdy);

    // cvReleaseMat(&imgYpatch);
    // cvReleaseMat(&imgUpatch);
    // cvReleaseMat(&imgVpatch);
    // cvReleaseMat(&mask);
  }
    
  m_initialized = 1;
}

void CvParticleFilter::observe()
{
  if (!initialized()) {return;}
  
  // draw new particles
  cvParticleTransition( m_particle );

  // measurement
// CV_TIMER_START();
  if ( !m_observer.measure(m_particle) ) {
    fprintf(stderr, "ERROR: particle measurement error!\n");
    return;
  }
// CV_TIMER_SHOW();

  int maxp_id = cvParticleGetMax( m_particle );
  CvBox2D box = cvParticleStateGet( m_particle, maxp_id );
  // fprintf(stderr, "center: %.2f,%.2f\n", box.center.x, box.center.y);
  m_window = box;

  // show best fit particle
  {
    float warp_p_data[3]={
      cos(box.angle/180.*CV_PI),
      box.center.x-box.size.width/2.,
      box.center.y-box.size.height/2.
    };
    CvMat warp_p = cvMat(3,1,CV_32F,warp_p_data);
    // m_observer.learn(&warp_p);

    // CvMat * imgYpatch=cvCreateMat(box.size.height,box.size.width,CV_8U);
    // icvWarp(m_imgY, imgYpatch, &warp_p);
    // CV_SHOW(imgYpatch);
    // cvReleaseMat(&imgYpatch);
  }

  // normalize
  cvParticleNormalize( m_particle);

  // resampling
  cvParticleResample( m_particle );
}
