/**
 * @file   main25_levelset2.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Thu Jan 17 13:07:00 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"

int main(int argc, char * argv[])
{
#if 1
  IplImage * img =
      cvLoadImage(argc>1?argv[1]:"../data/twocells.bmp", 0); 
  //IplImage * img = cvLoadImage("../data/palm007.png", 0); 
  CvSize imsize = cvGetSize(img); int nr = imsize.height,nc = imsize.width;
  CvLevelSetTracker * lstracker =
      cvCreateLevelSetTracker(imsize, CV_32F);

  CvRect rois[]={cvRect(nc/4,nr/4,nc/2,nr/2)};
  int niters;
  
  CV_TIMER_START();
  cvLevelSetPrepare(lstracker, img, rois, 1, 15);
  niters = cvLevelSetUpdate(lstracker,
                            (argc>2)?atol(argv[2]):5.f,          // timestep
                            0.2f/((argc>2)?atol(argv[2]):5.f),   // mu
                            5.f,                                 // lambda
                            (argc>3)?atol(argv[3]):1.5f,         // alpha
                            5, 100);            // inner and outer maxiter
  CV_TIMER_SHOW();

  fprintf(stderr, "niter: %d\n", niters);
  CvMat * bw = cvCreateMat(nr, nc, CV_8U);
  cvCmpS(lstracker->phi, 0, bw, CV_CMP_LT);
  cvCopy(img, bw, bw);
  cvShowImage("Test", bw); CV_WAIT();
  cvReleaseMat(&bw);
  cvReleaseLevelSetTracker(&lstracker);
#else
  IplImage * img = cvLoadImage("../data/gourd.bmp", 0);
  CvSize imsize = cvGetSize(img);
  CvLevelSetTracker * lstracker =
      cvCreateLevelSetTracker(imsize, CV_32F);

  CvRect rois[]={cvRect(20,25,5,10),cvRect(40,25,10,10)};
  
  cvLevelSetPrepare(lstracker, img, rois, 2);
  cvLevelSetUpdate(lstracker, 1.f, -3.f, 5, 100);
  cvReleaseLevelSetTracker(&lstracker);
#endif
  return 0;
}
