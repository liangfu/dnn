/**
 * @file   cvshapeprior.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Mar  7 17:48:01 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_SHAPE_PRIOR_H__
#define __CV_SHAPE_PRIOR_H__

#include "cvext_c.h"
#include "cvclassifier4ls.h"
#include "cvlevelset.h"

class CV_EXPORTS CvShapePriorData
{
  CvMat * meanshape;
  CvMat * mean;
  CvMat * pc;
  // CvMat * latent;
protected:
  void initialize();
public:
  CvShapePriorData():
      meanshape(0), mean(0), pc(0)// , latent(0)
  {
    initialize();
  }

  ~CvShapePriorData()
  {
    if (meanshape) {cvReleaseMat(&meanshape); meanshape=0;}
    if (mean) {cvReleaseMat(&mean); mean=0;}
    if (pc) {cvReleaseMat(&pc); pc=0;}
    // if (latent) {cvReleaseMat(&latent); latent=0;}
  }

  inline CvMat * get_meanshape(){return meanshape;}
  inline CvMat * get_mean(){return mean;}
  inline CvMat * get_pc(){return pc;}
  // inline CvMat * get_latent(){return latent;}
};

class CvShapePrior
{
  CvShapePriorData m_data;
  CvMat * meanshape;
  CvMat * mean;
  CvMat * pc;

  CvMat * term0;
  CvMat * term1;

  CvMat * proj;

  CvClassifier4Levelsets m_classifier;
  int m_status;
 public:
  CvShapePrior():
      //m_data(),
      meanshape(0), mean(0), pc(0),
      term0(0), term1(0),
      proj(0), //m_classifier(),
      m_status(-1)
  {
    meanshape=m_data.get_meanshape();
    mean=m_data.get_mean();
    pc=m_data.get_pc();
    proj = cvCreateMat(1, pc->cols, CV_32F);
    assert(meanshape);
    assert(mean);
    assert(pc);
  }
  ~CvShapePrior()
  {
    if (term0) {cvReleaseMat(&term0); term0=0;}
    if (term1) {cvReleaseMat(&term0); term0=0;}
    if (proj) {cvReleaseMat(&proj); proj=0;}
  }

  int initialized() {return ((meanshape!=NULL)&&(mean!=NULL)&&(pc!=NULL));}
  
  CvMat * shapeterm0(CvMat * hv);
  CvMat * shapeterm1(CvMat * hv);

  CvMat * classify(CvMat * bw)
  {
    CvMat * phi0 = cvCreateMat(bw->rows, bw->cols, CV_32F);
    icvInitializeLevelSet(bw,phi0);
    shapeterm1(phi0);
    cvReleaseMat(&phi0);
    return term1;//m_status;
  }

  int status() { return m_status; }
};

#endif // __CV_SHAPE_PRIOR_H__
