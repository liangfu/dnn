/**
 * @file   cvsgkfd.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 11:20:01 2013
 * 
 * @brief  sparse greedy kernel fisher discriminant
 * 
 * 
 */

#ifndef __CV_SGKFD_H__
#define __CV_SGKFD_H__

#include "cvext_c.h"

class CvSGKFD
{
 protected:
  int m_initialized;
  
  CvMat * alpha;
  CvMat * X;
  float b;

  // derive this class and override this function
  virtual int initialize()=0; 

 public:
  CvSGKFD():m_initialized(0),alpha(0),X(0),b(0){}

  ~CvSGKFD(){
    if (alpha){cvReleaseMat(&alpha);alpha=0;}
    if (X){cvReleaseMat(&X);X=0;}
  }

  int initialized(){ return (m_initialized && alpha && X); }

  /** 
   * inference with sparse greedy kernel fisher discriminant
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

#endif // __CV_SGKFD_H__
