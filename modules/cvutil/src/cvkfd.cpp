/**
 * @file   cvkfd.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 29 11:19:39 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvkfd.h"

int CvKFD::predict(CvMat * X_test, CvMat * response)
{
  if (!initialized()){LOGE("CvKFD isn't initialized!");return -1;}
  int N0=X->rows;
  int N1=X_test->rows;
  assert( (((response->cols==N1)&&(response->rows==2))||
           ((response->rows==N1)&&(response->cols==2))) &&
          (CV_MAT_TYPE(response->type)==CV_32F) );
  CvMat * K_test = cvCreateMat(N1,N0,CV_32F);
  CvMat * L_test = cvCreateMat(N1,2 ,CV_32F);
  icvGaussianKernel(X,X_test,K_test);
  cvMatMul(K_test,alpha,L_test);
  // cvScale(L_test,response,10.);
  {
    float * lptr = L_test->data.fl;
    float * rptr = response->data.fl;
    int i,j,lstep=L_test->step/sizeof(float),
        rstep=response->step/sizeof(float);
    for (i=0;i<N1;i++,lptr+=lstep,rptr+=rstep){
      rptr[0]=lptr[0]*10.;rptr[1]=-lptr[0]*10.;
    }
  }
  cvReleaseMat(&K_test);
  cvReleaseMat(&L_test);
  return 1;
}

