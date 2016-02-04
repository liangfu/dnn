/**
 * @file   cvlda.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Apr  2 08:50:41 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_LDA_H__
#define __CV_LDA_H__

#include "cvext_c.h"

class CV_EXPORTS CvLDA
{
protected:
  CvMat * W;
  CvMat * PRIOR;
public:
  CvLDA():W(0),PRIOR(0){}
  ~CvLDA(){
    if (W!=NULL){ cvReleaseMat(&W); W=NULL; }
    if (PRIOR!=NULL){ cvReleaseMat(&PRIOR); PRIOR=NULL; }
  }

  int train(CvMat * train_data, CvMat * response);
  
  int predict(CvMat * sample, CvMat * result);
  int predict_withprior(CvMat * sample, CvMat * result);
};

#endif // __CV_LDA_H__
