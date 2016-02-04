/**
 * @file   main19_invcomp.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Dec 24 16:10:59 2012
 * 
 * @brief  
 * 
 * 
 */

#include "highgui.h"
#include "cvinvcomp_.h"

int main(int argc, char * argv[])
{
  // IplImage * img = cvLoadImage("../data/takeo.pgm", 0);
  IplImage * img2 = cvLoadImage("../data/111.png", 0);
  IplImage * img3 = cvLoadImage("../data/222.png", 0);
  CvMat mat_stub, * mat, submat_stub, * submat, * init_p;
  CvMat * mat32f, * submat32f;
  const int maxiter=200;
  CvInvComp invcomp;

  mat = cvCloneMat(cvGetMat(img3, &mat_stub));
  // submat = cvCloneMat(cvGetSubRect(mat, &submat_stub,
  //                                  cvRect(50, 90, 50, 50)));
  submat = cvCloneMat(cvGetSubRect(img2, &submat_stub,
                                   cvRect(61, 120, 36, 42)));

  mat32f = cvCreateMat(mat->rows, mat->cols, CV_32F);
  submat32f = cvCreateMat(submat->rows, submat->cols, CV_32F);

  cvConvert(mat, mat32f);
  cvConvert(submat, submat32f);

  init_p = cvCreateMat(2, 3, CV_32F); 
  init_p->data.fl[0]=1;
  init_p->data.fl[1]=0;
  // init_p->data.fl[2]=55;
  // init_p->data.fl[3]=0;
  // init_p->data.fl[4]=1;
  // init_p->data.fl[5]=95;
  init_p->data.fl[2]=55;
  init_p->data.fl[3]=0;
  init_p->data.fl[4]=1;
  init_p->data.fl[5]=120;

#if 1
  invcomp.affine_fa(mat32f, submat32f, init_p, maxiter, 600);
#else
  invcomp.affine_ic(mat32f, submat32f, init_p, maxiter, 600);
#endif

  cvReleaseMat(&mat);
  cvReleaseMat(&submat);
  
  return 0;
}
