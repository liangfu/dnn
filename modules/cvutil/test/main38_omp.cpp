/**
 * @file   main38_omp.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Tue Jul 23 09:09:44 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvsparsecoding.h"
#include "cvtimer.h"

int main()
{
  float D_data[6]={.5,sqrt(3.)*.5,1,0,-1./sqrt(2.),-1./sqrt(2.)};
  float y_data[2]={1,.5};
  CvMat D = cvMat(3,2,CV_32F,D_data);
  // CvMat * D_T = cvCreateMat(2,3,CV_32F);
  // cvTranspose(&D,D_T);
  CvMat y = cvMat(2,1,CV_32F,y_data);
  CvMat * x = cvCreateMat(3,1,CV_32F);
  int maxiter = 10;
  float epsilon = .02;

CV_TIMER_START();
  if (!icvOrthogonalMatchingPursuit(&D, &y, x, maxiter, epsilon)){
    fprintf(stderr,"ERROR: fail to perform OMP algorithm.\n");
  }
CV_TIMER_SHOW();
  CvMat * approx = cvCreateMat(D.cols,x->cols,CV_32F);
  cvGEMM(&D,x,1,NULL,1,approx,CV_GEMM_A_T);
  cvPrintf(stderr, "%.2f,", approx);
  cvReleaseMat(&approx);

  cvReleaseMat(&x);
  // cvReleaseMat(&D_T);
  return 0;
}
