/**
 * @file   main06_handseg.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Thu Dec 06 17:10:41 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"

int main()
{
  cvNamedWindow("Test");
  char fname[1024];
  char * fname_sub[]={"palm_jc.png","palm_jiang.png","palm_jason.png"};
  char dirname[]={"../data/"};

  for (int i = 0; i <= 3; i++)
  {
	strcpy(fname, dirname); strcat(fname, fname_sub[i]);
    IplImage * img = cvLoadImage(fname, 0/*grayscale*/); assert(img);

	int nrows = img->height;
	int ncols = img->width;

	CvMat * toSource = cvCreateMat(nrows, ncols, CV_32F);
	CvMat * toTarget = cvCreateMat(nrows, ncols, CV_32F);

	CvMat * Bterm = cvCreateMat(nrows, ncols, CV_32F); // 

    CvMaxFlowSegmentation<CvGraph32f> engine(img);
    // engine.set_regional_term(toSource, toTarget);
    // engine.set_boundary_term(Bterm);
    // engine.maxflow();

	// IplImage * imgResult = cvCreateImage(cvSize(ncols, nrows), IPL_DEPTH_8U, 1);
    // engine.get_segmentation(imgResult);
	// cvShowImage("Test", imgResult); CV_WAIT();
	// cvReleaseImageEx(imgResult);

    cvShowImage("Test", img); CV_WAIT();
  }
  cvDestroyWindow("Test");
  return 0;
}

