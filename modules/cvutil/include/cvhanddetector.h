/**
 * @file   cvwavinghanddetector.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Feb  7 14:38:37 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_WAVING_HAND_DETECTOR_H__
#define __CV_WAVING_HAND_DETECTOR_H__

#include "cvext_c.h"

class CvWavingHandDetector
{
  struct {
    char ori;                            // current orientaion
    int fid;                              // how many continous wave
    CvBox2D roi;
    int con;
    int nth;
  } states[32];

  int iter;

 public:

  CvWavingHandDetector()
  {
    reset();
  }

  inline void reset ()
  {
    iter=0;
    memset(states, 0, sizeof(states));
  }
  
  void push(const char _ori, const int _fid, const CvBox2D _roi);

  int valid();                                  // criteria for validation

  inline CvBox2D get_roi()
  {
    return states[(iter>0)?(iter-1):0].roi;
  }

  inline void set_roi(const CvBox2D & _roi) {
    states[(iter>0)?(iter-1):0].roi=_roi;
  }
};

class CV_EXPORTS CvHandDetector
{
  int m_initialized;
  CvMat * edgemap;
  CvMat * gauss;

  CvMemStorage * storage;
  // CvHaarClassifierCascade * cascade;
  void * cascade;
  
  // static CvMat * rawread(char * fn)
  // {
  //   FILE * fp = fopen(fn, "r");
  //   if (!fp) {
  //     fprintf(stderr, "ERROR: can't load %s !!\n", fn); return 0;
  //   }
  //   int nr,nc,i,j;
  //   fscanf(fp, "%d %d\n", &nr, &nc);
  //   CvMat * mat = cvCreateMat(nr, nc, CV_32F);
  //   int step=mat->step/sizeof(float);
  //   float * fptr=mat->data.fl;
  //   for (i=0;i<nr;i++) {
  //     for (j=0;j<nc;j++) {
  //       fscanf(fp, "%f ", &fptr[j]);
  //     }
  //     fptr+=step;
  //   }
  //   fclose(fp);
  //   return mat;
  // }

public:
  // enum CvDetectorType {CV_MOTION=1,CV_CHAMFER=2};
  CvHandDetector(): // CvDetectorType _type
      m_initialized(0),edgemap(NULL),gauss(NULL),
      storage(0),cascade(0)
  {
    storage = cvCreateMemStorage();
    // cascade =
    //     (CvHaarClassifierCascade*)cvLoad(
    //         "../data/haarcascade_palm25.xml", 0, 0, 0 );
            // "../data/haarcascade_palm32.xml", 0, 0, 0 );
  } 
  ~CvHandDetector()
  {
    if (edgemap) { cvReleaseMat(&edgemap);edgemap=NULL;}
    if (gauss)   { cvReleaseMat(&gauss  );gauss  =NULL;}
    if (storage) { cvReleaseMemStorage(&storage); storage=NULL;}
  }

  inline int initialized() { return m_initialized; }
  
  // preload parameters
  // inline int load(char ** filenames, int nfiles)
  // {
  //   assert(m_initialized==0);
  //   assert(nfiles==2);
  //   edgemap=rawread(filenames[0]); if (edgemap==0) {return 0;}
  //   gauss  =rawread(filenames[1]); if (gauss  ==0) {return 0;}
  //   // cvShowImageEx("Test", edgemap, CV_CM_GRAY); CV_WAIT();
  //   // cvShowImageEx("Test", gauss, CV_CM_GRAY); CV_WAIT();
  //   m_initialized=1;
  //   return 1;
  // }

  int detect(CvMat * grayimg,
             CvPoint2D32f & center, float & size, float & score);

  int cascade_detect(CvMat * grayimg, 
                     CvPoint2D32f & center, float & size)
  {
    if (!cascade){fprintf(stderr, "ERROR: cascade detector not initialized!\n");return -1;}
    cvClearMemStorage(storage);
    CvSeq * objects =
#if 0
        cvHaarDetectObjects( grayimg, cascade, storage);//,
                             // 1.1, 2, 0, cvSize(25, 25) );
#else
	0;
#endif
    if (objects)
    {
      if (objects->total<1) {return 0;}
      CvRect * roi = (CvRect*)cvGetSeqElem(objects, 0);
      size = (roi->width+roi->height)/2.;
      if (size>50) {return 0;} // maxsize bound
      center = cvPoint2D32f(roi->x+size/2., roi->y+size/2.);
      return 1; // target detected
    }else{
      return 0; // target not found
    }
  }
};

#endif // __CV_WAVING_HAND_DETECTOR_H__
