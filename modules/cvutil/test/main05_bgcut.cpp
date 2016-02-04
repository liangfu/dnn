/**
 * @file   main05_bgcut.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Thu Dec 06 15:23:30 2012
 * 
 * @brief  background/foreground segmentation test
 * 
 * 
 */

#include "cvext.h"

int main()
{
  cvNamedWindow("Test");
  char fname[CV_MAXSTRLEN];char fname2[CV_MAXSTRLEN];
  for (int i = 2; i <= 5; i++)
  {
    sprintf(fname, "../data/skin%d.png", i);
    IplImage * img = cvLoadImage(fname, 0/*grayscale*/); assert(img);

#if 1
	int nrows = img->height;
	int ncols = img->width;

	CvMat * toSource = cvCreateMat(nrows, ncols, CV_32F);
	CvMat * toTarget = cvCreateMat(nrows, ncols, CV_32F);

	CvMat * Bterm = cvCreateMat(nrows, ncols, CV_32F); // 

    CvMaxFlowSegmentation<CvGraph32f> engine(img);
    engine.set_regional_term(toSource, toTarget);
    engine.set_boundary_term(Bterm);
    engine.maxflow();

	IplImage * imgResult = cvCreateImage(cvSize(ncols, nrows), IPL_DEPTH_8U, 1);
    engine.get_segmentation(imgResult);
	cvShowImage("Test", imgResult); CV_WAIT();
	cvReleaseImageEx(imgResult);
#endif    
    cvShowImage("Test", img); CV_WAIT();
  }
  cvDestroyWindow("Test");
  return 0;
}

