#ifndef __CV_PARTICLE_OBSERVE_H__
#define __CV_PARTICLE_OBSERVE_H__

#include "cvext_c.h"
#include "cvparticlebase.h"
#include "cvtracker.h"
#include "cvlda4hog.h"

/**
 * Set a state to a particle filter structure
 *
 * @param state     A CvRect32f structure
 * @param p         particle filter struct
 * @param p_id      particle id
 * @return void
 */
inline void cvParticleStateSet( CvParticle* p, int p_id, CvBox2D state )
{
  CvMat * state_mat, hdr; 
  state_mat = cvGetCol( p->particles, &hdr, p_id );
  assert(p->num_states==5);
  assert(CV_MAT_TYPE(p->particles->type)==CV_32F);

  float halfw = state.size.width/2.f;
  float halfh = state.size.height/2.f;
  CV_MAT_ELEM(*state_mat, float, 0, 0) = state.center.x-halfw;
  CV_MAT_ELEM(*state_mat, float, 1, 0) = state.center.y-halfh;
  CV_MAT_ELEM(*state_mat, float, 2, 0) = state.size.width;
  CV_MAT_ELEM(*state_mat, float, 3, 0) = state.size.height;
  CV_MAT_ELEM(*state_mat, float, 4, 0) = state.angle;

  //fprintf(stderr, "center: %.2f,%.2f\n", state.center.x, state.center.y);
}

inline CvBox2D cvParticleStateGet( CvParticle * p, int p_id )
{
  CvMat * state_mat, hdr; CvBox2D box;
  state_mat = cvGetCol( p->particles, &hdr, p_id );

  box.size.width  = CV_MAT_ELEM(*state_mat, float, 2, 0);
  box.size.height = CV_MAT_ELEM(*state_mat, float, 3, 0);
  float halfw = box.size.width/2.f;
  float halfh = box.size.height/2.f;
  box.center.x = CV_MAT_ELEM(*state_mat, float, 0, 0)+halfw;
  box.center.y = CV_MAT_ELEM(*state_mat, float, 1, 0)+halfh;
  box.angle = CV_MAT_ELEM(*state_mat, float, 4, 0);

  // fprintf(stderr, "center: %.2f,%.2f\n", box.center.x, box.center.y);
  return box;
}

CVAPI(void)
cvParticleStateConfig( CvParticle * p, const CvSize imsize, CvBox2D std );

//-----------------------------------------------------------------------
// PARTICLE OBSERVATION
//-----------------------------------------------------------------------
class CV_EXPORTS CvParticleObserve
{
  CvGenericTracker & m_tracker;
  CvLDA4HOG m_lda4hog;
  CvMat *& imgY;
  CvMat *& imgU;
  CvMat *& imgV;
  
  int m_initialized;
  int m_status;

  int hogsizes[3];
  CvMatND * ref0hog; 
  CvMatND * ref1hog; 
  CvMatND * currhog;

  CvMat * magni_full;
  CvMat * angle_full; 
  
protected:
  int nbins;
  int ngrids;

public:
  CvMatND * m_reffhist;
  CvMatND * m_refbhist;
  CvMatND * m_currhist;

  CvParticleObserve(CvGenericTracker & tracker):
      m_tracker(tracker),
      imgY(tracker.m_imgY),imgU(tracker.m_imgU),imgV(tracker.m_imgV),
      m_initialized(false),
      m_status(-1),
      ref0hog(NULL),ref1hog(NULL),currhog(NULL),
      magni_full(NULL),angle_full(NULL),
      nbins(8),ngrids(1),
      m_reffhist(NULL),m_refbhist(NULL),m_currhist(NULL)
  {
    int histsizes[3]={nbins,nbins,nbins};
    m_currhist = cvCreateMatND(3,histsizes,CV_32F);
    m_reffhist = cvCreateMatND(3,histsizes,CV_32F);
    m_refbhist = cvCreateMatND(3,histsizes,CV_32F);

    hogsizes[0]=7;
    hogsizes[1]=6;
    hogsizes[2]=9;
    if (!ref0hog) { ref0hog = cvCreateMatND(3, hogsizes, CV_32F); }
    if (!ref1hog) { ref1hog = cvCreateMatND(3, hogsizes, CV_32F); }
    if (!currhog) { currhog = cvCreateMatND(3, hogsizes, CV_32F); }
  }

  ~CvParticleObserve()
  {
    clear();
  }

  void clear()
  {
    if (ref0hog) { cvReleaseMatND(&ref0hog); ref0hog=NULL; }
    if (ref1hog) { cvReleaseMatND(&ref1hog); ref1hog=NULL; }
    if (currhog) { cvReleaseMatND(&currhog); currhog=NULL; }
    if (m_reffhist) { cvReleaseMatND(&m_reffhist); m_reffhist=NULL; }
    if (m_refbhist) { cvReleaseMatND(&m_refbhist); m_refbhist=NULL; }
    if (m_currhist) { cvReleaseMatND(&m_currhist); m_currhist=NULL; }
  }

  void load()
  {
    // no static data required !!
  }

  // void update(CvMat * imgYpatch, CvMat * imgUpatch, CvMat * imgVpatch,
  //             CvMat * submask=0);
  void initialize(CvRect roi);
  int initialized() { return m_initialized; }
  inline int status() { return m_status; }

  int measure(CvParticle * particle);
  
};

#endif // __CV_PARTICLE_OBSERVE_H__
