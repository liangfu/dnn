/**
 * @file   cvkfd.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 11:20:01 2013
 * 
 * @brief  kernel fisher discriminant
 * 
 * 
 */

#ifndef __CV_KFD_H__
#define __CV_KFD_H__

#include "cvext_c.h"

class CvKFD
{
 protected:
  int m_initialized;
  
  CvMat * alpha;
  CvMat * X;

  // derive this class and override this function
  virtual int initialize()=0; 

 public:
  CvKFD():m_initialized(0),alpha(0),X(0){}

  ~CvKFD(){
    if (alpha){cvReleaseMat(&alpha);alpha=0;}
    if (X){cvReleaseMat(&X);X=0;}
  }

  int initialized(){ return (m_initialized && alpha && X); }

  /** 
   * inference with kernel fisher discriminant
   * 
   * @param X_test    in: NxM matrix as test samples
   * @param response  in: Nx2 matrix as final result 
   * 
   * @return out: status code,
   *              negative values indicate errors,
   *              positive values indicate no error
   */
  int predict(CvMat * X_test, CvMat * response);
};

#endif // __CV_KFD_H__
