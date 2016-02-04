/**
 * @file   cvfacedetector.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jul  8 15:57:22 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_STAGED_DETECTOR_HAAR_H__
#define __CV_STAGED_DETECTOR_HAAR_H__

#include "cvext_c.h"
#include <vector>

// final classifier design :
// winsize - configure
// {thresh, polarity}, {x, y, w, h, wt}x3  => 2+5x3 => 17 elements

// 14 stages: .95,.99
//0,1,8,21,33,42,62,97,148,267,308,339,361,449,986
CvMat * get_haarclassifier_face20_v0();
// 10 stages, .85,.96
//0,1,2,10,29,73,103,160,196,201,316
CvMat * get_haarclassifier_face20_v1();

class CV_EXPORTS CvStagedDetectorHaar
{
  int m_initialized;
  CvMat * features;
  CvMat * evalres_precomp[2];
  int evalres_precomputed;
  CvMat * weights;
  std::vector<int> selected;
  int weights_initialized;
  int validate(int ni, double & fi, double & di);
  int adjust(int ni, double dtar, double & fi, double & di);

public:
  CvStagedDetectorHaar():
      m_initialized(false),
      features(NULL),evalres_precomputed(0),
      weights(0),weights_initialized(0)
  {
    evalres_precomp[0]=0;evalres_precomp[1]=0;
	selected.clear();
  }

  ~CvStagedDetectorHaar()
  {
    if (features) { cvReleaseMat(&features); }
#ifdef CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
    for (i=0;i<2;i++){
      if (evalres_precomp[i]){cvReleaseMat(&evalres_precomp[i]);}
    }
#endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
    if (weights) { cvReleaseMat(&weights); }
  }

  int detect(CvMat * img, CvRect roi[]);
  int train(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
            int maxiter=8000, int startiter=0);
  int cascadetrain(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
                   double fper=.7, double dper=.94, double ftarget=1e-5); 
};

#endif // __CV_STAGED_DETECTOR_HAAR_H__
