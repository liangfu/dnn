/**
 * @file   main39_bp.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Aug  2 16:06:30 2013
 * 
 * @brief  test case for basis pursuit 
 * 
 * 
 */
#include "cvsparsecoding.h"
#include "cvtimer.h"

int main()
{
  float D_data[6]={
    // .5,sqrt(3.)*.5,
    // 1,0,
    // -1./sqrt(2.),-1./sqrt(2.)
    .5,1,-1./sqrt(2.),sqrt(3.)*.5,0,-1./sqrt(2.)
  };
  float y_data[2]={1,.5};
  float x_data[3],approx_data[2];
  CvMat D = cvMat(2,3,CV_32F,D_data);
  CvMat y = cvMat(2,1,CV_32F,y_data);
  CvMat x = cvMat(3,1,CV_32F,x_data);
  CvMat approx = cvMat(2,1,CV_32F,approx_data);
  int maxiter = 3;

CV_TIMER_START();
  if (!icvBasisPursuit(&D, &y, &x, maxiter)){
    fprintf(stderr,"ERROR: fail to perform OMP algorithm.\n");
  }
CV_TIMER_SHOW();
  cvGEMM(&D,&x,1,NULL,1,&approx,0);
  cvPrintf(stderr, "%.4f,", &x);
  cvPrintf(stderr, "%.4f,", &approx);

  return 0;
}
