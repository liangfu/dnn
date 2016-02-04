/**
 * @file   main40_facecoder.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Aug  5 14:22:27 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvfacecoder.h"
#include "cvtimer.h"

int main()
{
  CvFaceCoderSRC facecoder;
  facecoder.config("/home/andrew/workspace");
  int i;

  const char * imglist[]={
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E+00.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E-20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E+20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E-35.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E+45.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+000E+90.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-005E-10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-005E+10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+005E-10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+005E+10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-010E+00.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+010E+00.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-010E-20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+010E-20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-015E+20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+015E+20.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-020E-10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-020E+10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+020E-10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A+020E+10.pgm",
    "../dataset/CroppedYale/yaleB01/yaleB01_P00A-020E-40.pgm",
 
    // "../data/yale/yaleB01_P00A-005E-10 (copy).pgm",
    // "../data/yale/yaleB01_P00A-010E-20 (copy).pgm",
    // "../data/yale/yaleB01_P00A-130E+20.pgm",
    // "../data/yale/yaleB02_P00A-130E+20.pgm",
    // "../data/yale/yaleB03_P00A-130E+20.pgm",
    // "../data/yale/yaleB04_P00A-130E+20.pgm",
    "../data/yale/yaleB01_P00A-005E-10.pgm",
    "../data/yale/yaleB02_P00A-005E-10.pgm",
    "../data/yale/yaleB04_P00A-005E-10.pgm",
    ""
  };
  char name[32];

  // LEARNING
//   for (i=0;strlen(imglist[i])!=0;i++)
//   {
//   IplImage * raw = cvLoadImage(imglist[i],0);
//   CvMat subimg_stub;
//   CvMat * subimg = cvGetMat(raw,&subimg_stub);
//   CvMat * resized = cvCreateMat(64,56,CV_8U);
//   cvResize(subimg,resized);
// CV_TIMER_START();  
//   strncpy(name,imglist[i]+13,7);
//   // fprintf(stderr,"name: %s\n",name);
//   facecoder.train(resized,name);
// CV_TIMER_SHOW();  
//   cvReleaseMat(&resized);
//   cvReleaseImage(&raw);
//   }

  // RECOGNITION
  for (i=0;strlen(imglist[i])!=0;i++)
  {
  IplImage * raw = cvLoadImage(imglist[i],0);
  CvMat subimg_stub;
  CvMat * subimg = cvGetMat(raw,&subimg_stub);
  CvMat * resized = cvCreateMat(64,56,CV_8U); 
  cvResize(subimg,resized); CV_SHOW(resized);
CV_TIMER_START();  
  facecoder.predict(resized,name);
  fprintf(stderr,"predict: %s\n",name);;
CV_TIMER_SHOW();  
  cvReleaseMat(&resized);
  cvReleaseImage(&raw);
  }
  
  return 0;
}
