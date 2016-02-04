/**
 * @file   cvkfd4hog.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 13:43:07 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_SGKFD_FOR_HOG_H__
#define __CV_SGKFD_FOR_HOG_H__

#include "cvsgkfd.h"

class CV_EXPORTS CvSGKFD4HOG : public CvSGKFD
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

    alpha = cvCreateMat(N0,1,CV_32F);
    X = cvCreateMat(N0,M,CV_32F);
    memcpy(alpha->data.ptr,alpha_data,sizeof(float)*N0);
    memcpy(X->data.ptr,X_data,sizeof(float)*N0*M);
    b=%f;
    m_initialized=1;

    return 1;
  }
  
 public:
  CvSGKFD4HOG():CvSGKFD(){initialize();}
};

#endif // __CV_SGKFD_FOR_HOG_H__
