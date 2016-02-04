/**
 * @file   cvpfilter.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Sun Jan  6 14:33:35 2013
 * 
 * @brief  general particle filter implementation
 */

#ifndef __CV_EXT_PFILTER_H__
#define __CV_EXT_PFILTER_H__

#include "cvext_c.h"
#include "cvext.hpp"
#include "cvtracker.h"
#include "cvparticlebase.h"
#include "cvparticleobserve.h"
#include "cvchamfer.h"

//------------------------------------------------------------
// GENERAL PARTICLE FILTER IMPLEMENTATION
//------------------------------------------------------------
class CvParticleFilter
{
  CvGenericTracker & m_tracker;
  CvParticleObserve m_observer;
  CvMat *& m_imgY;
  CvMat *& m_imgU;
  CvMat *& m_imgV;

  CvParticle * m_particle;
  CvBox2D m_window;

  bool m_initialized;
  CvSize & m_imsize;

  int N_p;

 public:
  inline int initialized() {
    return m_initialized && m_observer.initialized();
  }

  CvParticleFilter(CvGenericTracker & tracker):
      m_tracker(tracker),
      m_observer(tracker),
      m_imgY(tracker.m_imgY),
      m_imgU(tracker.m_imgU),
      m_imgV(tracker.m_imgV),
	  m_particle(NULL),
      m_initialized(false),
      m_imsize(tracker.m_imsize),
      N_p(5)
  {
    // states, observes, uselog
    m_particle = cvCreateParticle( N_p , 200, 1 ); 
  }
  ~CvParticleFilter(){}

  // void config(float dx=1.f,float dy=1.f,
  //             float dw=.3f,float dh=.3f,float dtheta=0.5f)
  // void config(float dx,float dy,
  //             float dw,float dh,float dtheta)
  // {
  //   float lin = m_imsize.width/160.f;
  //   CvBox2D box = cvBox2D(dx*lin, dy*lin, dw*lin, dh*lin, dtheta);
  //   cvParticleStateConfig( m_particle, m_imsize, box);
  // }
  
  void initialize(CvRect roi);

  void observe();

  inline CvBox2D get_window() { return m_window; }
  inline int status() { return m_observer.status(); }
};

#endif // __CV_EXT_PFILTER_H__
