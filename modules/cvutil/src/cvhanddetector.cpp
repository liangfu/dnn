/**
 * @file   cvwavinghanddetector.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Feb  7 14:37:53 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvhanddetector.h"

void CvWavingHandDetector::push(const char _ori,
                                const int _fid,
                                const CvBox2D _roi)
{
  static int last_fid=0;
    
  if (valid()) {return;}

  if ( ((last_fid!=0) && ((_fid-1)!=last_fid)) )
  {
    last_fid=0; reset();
  }
  else if (iter==0)
  {
    last_fid = _fid;
    states[iter].ori=_ori;
    states[iter].fid=_fid;
    states[iter].roi=_roi;
    states[iter].con=1;
    states[iter].nth=1;
    iter++;
  }
  else if (iter<32)
  {
    last_fid = _fid;
    states[iter].ori=_ori;
    states[iter].fid=_fid;
    states[iter].roi=_roi;
    if (states[iter].ori!=states[iter-1].ori) // switch
    {
      if (states[iter-1].con<3) { reset(); return; }
      // else {
      //   fprintf(stderr, "=======switch at: %d", states[iter-1].con);
      // }
      states[iter].con = 1;
      states[iter].nth = states[iter-1].nth+1;
    }
    else                                      // continuous
    {
      if (states[iter-1].con>6) { reset(); return; } 
      states[iter].con = states[iter-1].con+1;
      states[iter].nth = states[iter-1].nth;
    }
    iter++;
  }
  else{ reset(); }
} 

int CvWavingHandDetector::valid()
{
  if (iter==0) {return 0;}
  else if (iter<16) {
    return 0;
  }
  // else if ( (states[iter-1].nth>=4) && (states[iter-1].con>=2)){
  else if ( (states[iter-1].nth>=4) && (states[iter-1].con>=2)){
  // else if ( (states[iter-1].nth>=4)&&(states[iter-1].con>=4) ) {
    if (0)
    {
      fprintf(stderr, "==============wave: ");
      for (int i = 1; i < iter+1; i++) {
        if (states[i].con<states[i-1].con)
          fprintf(stderr, "%d, ", states[i-1].con);
      }
      fprintf(stderr, "==============\n");
    }
    return 1;
  }else{return 0;}
}

int CvHandDetector::detect(CvMat * grayimg,
                           CvPoint2D32f & center,
                           float & size, float & score)
{
  int found=0;
  assert(initialized());
  int nr=grayimg->rows, nc=grayimg->cols;
  CvMat * cannyedge = cvCreateMat(nr,nc,CV_8U);
  CvMat * invcanny = cvCreateMat(nr,nc,CV_8U);
  CvMat * distmap = cvCreateMat(nr, nc, CV_32F);
  CvMat * scoremap = cvCreateMat(nr, nc, CV_32F);
  CvMat * mscore = cvCreateMat(nr+1-gauss->rows,nc+1-gauss->cols,CV_32F);
  double minval, maxval;CvPoint minloc,maxloc;

  cvCanny(grayimg, cannyedge, 5, 200, 3);
  cvSubRS(cannyedge, cvScalar(255), invcanny);
  cvDistTransform(invcanny, distmap);
  cvThreshold(distmap,distmap,20,20,CV_THRESH_TRUNC);

  cvFilter2D(distmap,scoremap,edgemap,
             cvPoint(edgemap->cols/2.,edgemap->rows/2.));
  {
    cvMinMaxLoc(scoremap, &minval, &maxval, &minloc, &maxloc);
    cvThreshold(scoremap,scoremap,minval+(maxval-minval)/20.f,
                minval+(maxval-minval)/20.f,CV_THRESH_TRUNC);
    cvSubRS(scoremap, cvScalar(minval+(maxval-minval)/20.f), scoremap);
    score=minval;
  }

  cvFilter2D(scoremap,scoremap,gauss,
             cvPoint(gauss->cols/2.,gauss->rows/2.));
  {
    cvMinMaxLoc(scoremap, &minval, &maxval, &minloc, &maxloc);
    cvThreshold(scoremap,scoremap,0,0,CV_THRESH_TOZERO);
  }

  fprintf(stderr, "\n%f,%f\n", score, maxval);
  if ((score>100)&&(score<110)&&(maxval>1500)){
    // cvCircle(scoremap,maxloc,20,cvScalarAll(2550),2);
    // cvShowImageEx("Test", scoremap, CV_CM_GRAY); CV_WAIT2(100);
    size=20;
    center=cvPointTo32f(maxloc);
    found=1;
  }

  cvReleaseMat(&invcanny);
  cvReleaseMat(&cannyedge);
  cvReleaseMat(&distmap);
  cvReleaseMat(&scoremap);
  cvReleaseMat(&mscore);
  return found;
}

