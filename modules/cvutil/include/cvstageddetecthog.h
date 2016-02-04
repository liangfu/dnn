/**
 * @file   cvstageddetectorhog.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Sep 10 15:19:05 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_STAGED_DETECTOR_HOG_H__
#define __CV_STAGED_DETECTOR_HOG_H__

#include "cvext_c.h"
#include "cvhog.h"

class CvStagedDetectorHOG
{
  int m_initialized;
  CvMat * m_features;
  CvMat * m_weights;
  int validate(int ni, double & fi, double & di);
  int adjust(int ni, double dtar, double & fi, double & di);

  // classifiers
  typedef int (CvStagedDetectorHOG::*trainfunctype)
  (CvMat **, int, CvMat **, int, int);
  int train_svm(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
                int iter){  return 1; }
  int train_ada(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
                int iter);

  int m_feature_precomputed;
  int feature_precompute_hog(CvMat ** posimgs, int npos,
                             CvMat ** negimgs, int nneg,
                             CvMat ** magni, CvMat ** angle);
  
 public:
  CvStagedDetectorHOG():
      m_initialized(false),m_features(NULL),m_weights(NULL),
      m_feature_precomputed(false)
  {
  }
  
  ~CvStagedDetectorHOG()
  {
    if (m_features){cvReleaseMat(&m_features);}
    if (m_weights ){cvReleaseMat(&m_weights);}
  }

  int initialized(){return m_initialized;}
  int detect(CvMat * img, CvRect ROIs[]);
  int cascadetrain(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
                   double fper=.5, double dper=.98, double ftarget=1e-5);
};

#endif // __CV_STAGED_DETECTOR_HOG_H__
