/**
 * @file   cvkfd4hog.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 13:43:07 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_KFD_FOR_HOG__
#define __CV_KFD_FOR_HOG__

#include "cvkfd.h"

class CV_EXPORTS CvKFD4HOG : public CvKFD
{
 protected:
  int initialize()
  {
    assert(!initialized());
    int M=%d,N0=%d;

    static float alpha_data[]={
      %s
    };

    static float X_data[]={
      %s
    };

    alpha = cvCreateMat(N0,2,CV_32F);
    X = cvCreateMat(N0,M,CV_32F);
    memcpy(alpha->data.ptr,alpha_data,sizeof(float)*N0*2);
    memcpy(X->data.ptr,X_data,sizeof(float)*N0*M);
    m_initialized=1;

    return 1;
  }
  
 public:
  CvKFD4HOG():CvKFD(){initialize();}
};

#endif // __CV_KFD_FOR_HOG__
