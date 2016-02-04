/**
 * @file   cvext_hand.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Dec 10 13:20:53 2012
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_EXT_HAND_H__
#define __CV_EXT_HAND_H__

#include "cvparticlefilter.h"
#include "cvlevelset.h"
#include "cvpwptracker.h"
#include "cvhanddetector.h"

class CV_EXPORTS CvHandTracker :
    public CvPWPTracker, 
    public CvParticleFilter
{
  IplImage * maskImage;
  IplImage * mask_prev;

  // CvWavingHandDetector m_validator;
  CvHandDetector m_detector;
  
  // int detect(CvRect roi);
  // int track();
  
public:
  int myTrack(int has_fg);//tzg
  void myUpdate();

public:
  CvHandTracker(CvGenericTracker & t):
      CvPWPTracker(t),
      CvParticleFilter(t),
      maskImage(NULL), mask_prev(NULL)
  {
    // char * filenames[] = {
    //   (char*)"../data/shapeprior-chamfer.raw",
    //   (char*)"../data/shapeprior-gaussian.raw"
    // };
    // m_detector.load(filenames,2);
  }

  ~CvHandTracker()
  {
    if ( maskImage ){ cvReleaseImage(&maskImage); maskImage=NULL; }
    if ( mask_prev ){ cvReleaseImage(&mask_prev); mask_prev=NULL; }
  }

  void update();

  // inline int valid () {
  //   return m_validator.valid();
  // }

  inline CvBox2D get_exwin() { return m_outerbox; }
  inline CvBox2D get_inwin() { return m_innerbox; }

  inline CvBox2D get_pfwin() { return CvParticleFilter::get_window(); }
  inline int status() { return CvParticleFilter::status(); }
  // inline CvRect get_window2() {
  //   CvRect roi=cvRect(0,0,0,0);
  //   if (phi)
  //   {
  //     int nr = phi->rows,nc=phi->cols;
  //     CvBox2D box = m_validator.get_roi();
  //     // keep center, modify size and rotation
  //     box.size.height=nr; box.size.width=nc; box.angle=0; 
  //     roi = cvBox2DToRect(box);
  //     roi.height=nr; roi.width=nc;
  //   }
  //   return roi;
  // }

  inline CvMat * get_segmentation() {return CvPWPTracker::bw;}
  inline int initialized() {return CvPWPTracker::initialized(); }
};

#endif // __CV_EXT_HAND_H__
