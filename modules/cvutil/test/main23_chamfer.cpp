/**
 * @file   main23_chamfer.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Wed Jan  9 11:13:51 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"
#include "cvxrcpalm000.h"

CvPoint chamfer(IplImage * sample, IplImage * mask);

int main(int argc, char * argv[])
{
  CvCapture * capture =
      cvCreateCameraCapture(0);
      // cvCreateFileCapture(argv[1]);
  // cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, 550);
  IplImage * resized = cvCreateImage(cvSize(320,240),8,3);
  IplImage * gray = cvCreateImage(cvSize(320,240),8,1);
  IplImage * yuv = cvCreateImage(cvSize(320, 240), 8, 3);
  IplImage * mask = cvCreateImage(cvSize(320,240),8,1);
  
  while (1)
  {
    IplImage * raw = cvQueryFrame(capture);
    /* IplImage * raw = cvLoadImage(argv[1], 1); */
    cvResize(raw, resized);
    cvCvtColor(resized, yuv, CV_BGR2HSV);
    cvInRangeS(yuv,
               cvScalar(0,0,0),
               cvScalar(85,255,255), mask);
               /* cvScalar(0,133,77), */
               /* cvScalar(255,173,127), mask); */
    cvSplit(yuv, 0,0,gray, 0);

    IplImage * dispImage = cvCloneImage(resized);
    CvPoint pt = chamfer(gray, mask);
    if (pt.x!=0) { cvCircle(dispImage, pt, 10, CV_RED, 2); } 
    // cvShowImage("Test", mask); CV_WAIT();
    cvShowImage("Test", dispImage); CV_WAIT2(10);
    cvReleaseImage(&dispImage);
  }

  cvReleaseImage(&resized);
  cvReleaseImage(&gray);
  cvReleaseImage(&yuv);
  cvReleaseImage(&mask);
  return 0;
}

CvPoint chamfer(IplImage * sample, IplImage * mask)
{
  CvMat A = cvMat(41, 36, CV_8U, xrcCVXRCPALM000);
  // IplImage * sample =
  //     cvLoadImage(argv[1], 0); // grayscale
  CvSize imsize = cvGetSize(sample);
  int nr=imsize.height, nc=imsize.width;

  CvMat * B = cvCreateMat(nr, nc, CV_8U);
  CvMat * bw = cvCreateMat(nr, nc, CV_8U);
  CvMat * dist = cvCreateMat(nr, nc, CV_32F);
  CvMat * score = cvCreateMat(nr, nc, CV_32F);

  {
    CvMat * dx = cvCreateMat(nr, nc, CV_16S);
    CvMat * dy = cvCreateMat(nr, nc, CV_16S);

    // cvSobel(sample, dx, 1, 0, 1);
    // cvSobel(sample, dy, 0, 1, 1);
    cvSobel(sample, dx, 1, 0, 3);
    cvSobel(sample, dy, 0, 1, 3);

    float a, b;
    int i, j;
    for (i = 0; i < nr; i++)
      for (j = 0; j < nc; j++)
      {
        a = CV_MAT_ELEM(*dx, short, i, j);
        b = CV_MAT_ELEM(*dy, short, i, j);
        CV_MAT_ELEM(*B, uchar, i, j) = sqrt(a*a+b*b);
      }

    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
  }

  cvCmpS(B, 20, bw, CV_CMP_LT);

  {
CV_TIMER_START();
  cvDistTransform(bw, dist);
CV_TIMER_SHOW();
  }

  CvPoint minloc, maxloc; double minval, maxval;
  {
    CvMat * kernel = cvCreateMat(A.rows, A.cols, CV_32F);
    cvSet(&A, cvScalar(1), &A);
    cvConvert(&A, kernel);
    cvSubRS(&A, cvScalar(1), &A);
    cvSet(kernel, cvScalar(-34./1442.), &A);
    cvFilter2D(dist, score, kernel, cvPoint(A.rows/2, A.cols/2));

    // select skin color region
    {
      CvMat * tmp = cvCloneMat(score);
      cvSet(score, cvScalar(255));
      cvCopy(tmp, score, mask);
      cvReleaseMat(&tmp);
    }
    
    cvMinMaxLoc(score, &minval, &maxval, &minloc, &maxloc);
    fprintf(stderr, "least score: %f\n", minval);

    // cvSubRS(score, cvScalar(maxval), score);
    double thres = (maxval-minval)/200.;
    cvThreshold(score, score, thres, 255, CV_THRESH_TRUNC);
    cvSubRS(score, cvScalar(thres), score);
    
    // cvShowImageEx("Test", kernel, CV_CM_GRAY); CV_WAIT();
    cvReleaseMat(&kernel);
  }
  
  // cvShowImageEx("Test", B, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", bw, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", dist, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", score, CV_CM_GRAY); CV_WAIT2(10);
  // cvShowImageEx("Test", score); CV_WAIT2(10);

  cvReleaseMat(&B);
  cvReleaseMat(&bw);
  cvReleaseMat(&dist);
  cvReleaseMat(&score);
  
  // if (minval<20){
  if (minval<-10){
    return minloc;
  }else{
    return cvPoint(0,0);
  }
}
