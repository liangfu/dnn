/**
 * @file   cvtracker.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Feb  6 16:26:05 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_TRACKER_H__
#define __CV_TRACKER_H__

#include "cvext_c.h"

/**
 * Example:
 *
 *  CvAbstractTracker tracker;
 *  CvCapture * capture = cv....;
 *
 *  while(1)
 *  {
 *    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter);
 *    IplImage * rawImage = cvQueryFrame(capture);
 *    // cvCvtColor(rawImage, grayImage, CV_BGR2GRAY);
 *    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}
 *    if (tracker.initialized()){
 *        tracker.update(rawImage);
 *    }else{
 *        tracker.initialize(rawImage);
 *    }
 *  }
 */
class CV_EXPORTS CvAbstractTracker
{
public:
  IplImage * m_prevColorImage;      // previous BGR color image
  IplImage * m_currColorImage;
  IplImage * m_nextColorImage;      // next .. (optional), NULL for unavail

  IplImage * m_prevImage;           // previous grayscale image
  CvMat    * m_currImage;           // current ..
  IplImage * m_nextImage;           // next .. (optional)

  // set during initialization
  bool m_initialized;
  CvSize m_imsize;

  int m_framecounter;

  int m_iSqrMultiplier; // parameters affacted by resolution - squared
  int m_iLinMultiplier; // parameters affacted by resolution - linear
  
public:
  CvAbstractTracker():
      m_prevColorImage(NULL),
      m_currColorImage(NULL),
      m_nextColorImage(NULL),
      m_prevImage(NULL),
      m_currImage(NULL),
      m_nextImage(NULL),
      m_initialized(false),
      m_imsize(cvSize(0,0)),
      m_framecounter(0),
      m_iSqrMultiplier(-1),m_iLinMultiplier(-1)
  {}

  ~CvAbstractTracker()
  {
    cvReleaseImageEx(m_prevColorImage);
    cvReleaseImageEx(m_currColorImage);
    cvReleaseImageEx(m_nextColorImage);
    cvReleaseImageEx(m_prevImage);
    cvReleaseMatEx(m_currImage);
    cvReleaseImageEx(m_nextImage);
  }

  /** 
   * Initialize tracker with first frame 
   * 
   * @param frame   IN: BGR colro image as first frame in traker
   * @param roi     IN: rectangle boundary of ROI, (optional)
   * @param region  IN: contour of ROI, an arbitary shape, (optional)
   * 
   * @return OUT: ERROR code, 0 for no error 
   */
  int initialize(
      const CvArr * frame,                          // color frame
      const CvRect roi CV_DEFAULT(cvRect(0,0,0,0)), // rectangular ROI
      const CvArr * region CV_DEFAULT(NULL)         // contour of ROI
                      )
  {
    CV_FUNCNAME("CvAbstractTracker::initialize");
    __BEGIN__;
    CV_ASSERT(m_initialized==false);
    m_imsize = cvGetSize(frame);
    m_iSqrMultiplier = (m_imsize.height/120)*(m_imsize.width/160);
    m_iLinMultiplier = m_imsize.height/120; //m_imsize.width/160
    
    CV_ASSERT(cvGetElemType(frame)==CV_8UC3); // color image as input
    CV_ASSERT(m_prevColorImage==NULL);        // not initialized

    if (m_prevColorImage==NULL){
      IplImage header;
      m_nextColorImage = cvCloneImage(cvGetImage(frame, &header));
      if (m_nextImage==NULL){
        m_nextImage = cvCreateImage(m_imsize, IPL_DEPTH_8U, 1);
      }
      cvCvtColor(m_nextColorImage, m_nextImage, CV_BGR2GRAY);
    }

    if (m_currColorImage==NULL){
      m_currColorImage = cvCloneImage(m_nextColorImage);
    }
    if (m_prevColorImage==NULL){
      m_prevColorImage = cvCloneImage(m_currColorImage);
    }

    // initialize grayscale images
    if (!m_prevImage) {
      m_prevImage=cvCreateImage(m_imsize, IPL_DEPTH_8U, 1);
    }
    if (!m_currImage) {
      // m_currImage=cvCreateImage(m_imsize, IPL_DEPTH_8U, 1);
      m_currImage=cvCreateMat(m_imsize.height,m_imsize.width, CV_8U);
    }
    if (!m_nextImage) {
      m_nextImage=cvCreateImage(m_imsize, IPL_DEPTH_8U, 1);
    }
    // convert to grayscale image
    cvCvtColor(m_prevColorImage, m_prevImage, CV_BGR2GRAY);
    cvCvtColor(m_currColorImage, m_currImage, CV_BGR2GRAY);
    cvCvtColor(m_nextColorImage, m_nextImage, CV_BGR2GRAY);
    
    __END__;
    m_initialized = 1;
    return CV_StsOk;
  }

  /** 
   * Update tracker data with a new frame 
   * 
   * @param frame IN: BGR color image as new input 
   * 
   * @return OUT: ERROR code, 0 for no error  
   */
  int update(const CvArr * frame)
  {
    CV_FUNCNAME("CvAbstractTracker::update");
    __BEGIN__;

    CV_ASSERT(m_initialized);
    m_framecounter++;

    cvCopy(m_currColorImage, m_prevColorImage);
    cvCopy(m_nextColorImage, m_currColorImage);
    cvCopy(frame, m_nextColorImage);

    cvCvtColor(m_prevColorImage, m_prevImage, CV_BGR2GRAY);
    cvCvtColor(m_currColorImage, m_currImage, CV_BGR2GRAY);
    cvCvtColor(m_nextColorImage, m_nextImage, CV_BGR2GRAY);
    __END__;
    return CV_StsOk;
  }

  inline bool initialized() { return m_initialized; }
};

class CV_EXPORTS CvGenericTracker : public CvAbstractTracker
{

 public:
  CvMat * m_imgYUV;
  CvMat * m_imgY;    // grayscale image
  CvMat * m_imgU;    // `Cr` channel
  CvMat * m_imgV;    // 'Cb' channel
  CvMat * dx, * dy;

 public:
  CvGenericTracker():
      CvAbstractTracker(),
      dx(NULL), dy(NULL),
      m_imgYUV(NULL), m_imgY(NULL), m_imgU(NULL),m_imgV(NULL)
  {
  }

  ~CvGenericTracker()
  {
    if (dx) { cvReleaseMat(&dx); dx=NULL; }
    if (dy) { cvReleaseMat(&dy); dy=NULL; }

    if (m_imgYUV) { cvReleaseMat(&m_imgYUV); m_imgYUV=NULL; }
    if (m_imgY)   { cvReleaseMat(&m_imgY);   m_imgY=NULL; }
    if (m_imgU)   { cvReleaseMat(&m_imgU);   m_imgU=NULL; }
    if (m_imgV)   { cvReleaseMat(&m_imgV);   m_imgV=NULL; }
  }
  
  int initialize(const CvArr * frame) // color frame
  {
    if ( m_initialized ){
      fprintf(stderr, "WARNING: tracker already initialized!\n");
      return 0;
    }
    CvAbstractTracker::initialize(frame);
    m_initialized = 0;                          // undo initialization flag

    int nr=m_imsize.height,nc=m_imsize.width;
    if (!m_imgYUV){ m_imgYUV = cvCreateMat(nr, nc, CV_8UC3); }
    if (!m_imgY){ m_imgY = cvCreateMat(nr, nc, CV_8U); }
    if (!m_imgU){ m_imgU = cvCreateMat(nr, nc, CV_8U); }
    if (!m_imgV){ m_imgV = cvCreateMat(nr, nc, CV_8U); }
    cvCvtColor(m_currColorImage, m_imgYUV, CV_BGR2YUV);
    cvSplit(m_imgYUV, m_imgY, m_imgU, m_imgV, NULL);

    if (!dx) { dx=cvCreateMat(nr,nc,CV_16S); }
    if (!dy) { dy=cvCreateMat(nr,nc,CV_16S); }
    cvSobel(m_imgY,dx,1,0,1);
    cvSobel(m_imgY,dy,0,1,1);

    m_initialized = 1;                          // redo initialization flag
	return 1;
  }

  int update(const CvArr * frame) // color frame
  {
    if ( !m_initialized ){
      fprintf(stderr, "ERROR: tracker not initialized!\n");
      return 0;
    }
    CvAbstractTracker::update(frame);

    cvCvtColor(m_currColorImage, m_imgYUV, CV_BGR2YUV);
    cvSplit(m_imgYUV, m_imgY, m_imgU, m_imgV, NULL);
    
    cvSobel(m_imgY,dx,1,0,1);
    cvSobel(m_imgY,dy,0,1,1);

	return 1;
  }
};

#endif // __CV_TRACKER_H__
