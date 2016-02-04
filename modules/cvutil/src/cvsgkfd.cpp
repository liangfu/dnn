/**
 * @file   cvsgkfd.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 11:19:39 2013
 * 
 * @brief  sparse greedy kernel fisher discriminant
 * 
 * 
 */

#include "cvsgkfd.h"

int CvSGKFD::predict(CvMat * X_test, CvMat * response)
{
  if (!initialized()){LOGE("CvKFD isn't initialized!");return -1;}
  int N0=X->rows;
  int N1=X_test->rows;
  assert( (((response->cols==N1)&&(response->rows==1))||
           ((response->rows==N1)&&(response->cols==1))) &&
          (CV_MAT_TYPE(response->type)==CV_32F) );
  CvMat * K_test = cvCreateMat(N1,N0,CV_32F);
  CvMat * L_test = cvCreateMat(N1,alpha->cols,CV_32F);
  icvGaussianKernel(X,X_test,K_test);
  cvMatMul(K_test,alpha,L_test);
  cvAddS(L_test,cvScalar(b),response);
  cvReleaseMat(&K_test);
  cvReleaseMat(&L_test);
  return 1;
}
