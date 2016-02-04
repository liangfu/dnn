/**
 * @file   main28_shapereg.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Apr  1 10:03:42 2013
 * 
 * @brief  
 * 
 * 
 */


#include "cvshapedesc.h"

int main()
{
  const char * train_img_list[] = {
    "A-uniform15.pgm",
    // "A-uniform25-left-erode.pgm",
    // "A-uniform25-left.pgm",
    // "A-uniform25-orig.pgm",
    "A-uniform37.pgm",
    "A-uniform43-erode.pgm",
    "A-uniform43.pgm",
    "Five-uniform00-dilate.pgm",
    "Five-uniform04.pgm",
    "Five-uniform06-erode.pgm",
    "Five-uniform06.pgm",
    "Five-uniform07-dilate.pgm",
    "Five-uniform07.pgm",
    "Five-uniform09.pgm",
    "Five-uniform12.pgm",
    "Five-uniform17-erode.pgm",
    "Five-uniform17.pgm",
    "Five-uniform25.pgm",
    "Five-uniform40.pgm",
    "Five-uniform66.pgm",
    "Five-uniform73-dilate.pgm",
    "Five-uniform73.pgm"
  };
  // const int N[] = {2,10};
  const int N[] = {4,15};
  const int N_total = N[0]+N[1];
  int i;
  IplImage ** img = new IplImage*[N_total];
  CvMat ** imgdata = new CvMat*[N_total];
  CvMat * imgdata_stub = new CvMat[N_total];
  CvMat * labels = cvCreateMat(N_total, 1, CV_32S);

  for (i=0;i<N_total;i++)
  {
    img[i] = cvLoadImage(train_img_list[i], 0);
    imgdata[i] = cvGetMat(img[i], &imgdata_stub[i]);
    assert(CV_MAT_TYPE(imgdata[i]->type)==CV_8U);
  }
  for (i=0   ;i<N[0]     ;i++) {CV_MAT_ELEM(*labels, int, i, 0)=1;}
  for (i=N[0];i<N[0]+N[1];i++) {CV_MAT_ELEM(*labels, int, i, 0)=0;}
  
  CvFourierDescriptor fdesc(22);
  fdesc.train(imgdata, labels);

  CvMat * result = cvCreateMat(1,2,CV_32F);
  for (i=0;i<12;i++)
  {
    fdesc.predict(imgdata[i], result);
    cvPrintf(stderr, "%f,", result);
  }
  cvReleaseMat(&result);

  delete [] img;
  delete [] imgdata;
  delete [] imgdata_stub;
  cvReleaseMat(&labels);
  
  return 0;
}
