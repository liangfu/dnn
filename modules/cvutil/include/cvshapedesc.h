/**
 * @file   cvshapedesc.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Apr  1 11:16:05 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_SHAPE_DESC_H__
#define __CV_SHAPE_DESC_H__

#include "cvext_c.h"
// #include "cvlda.h"
#include "cvclassifier4fdesc.h"
//#include "ml.h"

CVAPI(void) cvExtractClosedContour(
    CvMat * bw, CvSeq * contour, CvMemStorage * storage);
CVAPI(void) cvExtractFourierDescriptor(
    CvMat * contour, CvMat * fdesc);

//------------------------------------------------------------
// a set of static functions for constructing Fourier Descriptor
// from shape, and use it for shape comparison
//------------------------------------------------------------
class CV_EXPORTS CvFourierDescriptor
{
  int fdsize;
  //CvNormalBayesClassifier m_classifier;
  CvClassifier4FourierDesc m_classifier;

public:
  CvFourierDescriptor(const int _fdsize){ fdsize=_fdsize; }
  ~CvFourierDescriptor(){}

  int train(CvMat ** train_data, CvMat * responses);
  int predict(CvMat * sample, CvMat * result);
};

#endif // __CV_SHAPE_DESC_H__

/** 
 * fourier descriptor extraction with normalization
 * 
 * @param contour   IN: contour 
 * @param fd        OUT: normalized fourier descriptor in size K
 */
/*
  static void extract(const CvArr * contour,       // 1xN CV_32FC2 
  CvArr * fdesc)               // 1xK CV_32FC2 
  {
  assert(contour); assert(fdesc);
  CvMat matContourHeader, matFDescHeader;
  CvMat * matContour = cvGetMat(contour, &matContourHeader);
  CvMat * matFDesc = cvGetMat(fdesc, &matFDescHeader); cvZero(matFDesc);
  assert(matContour->rows==1);
  assert(matFDesc->rows==1);

  const int conlen = matContour->cols;  // length of contour
  const int fdsize = matFDesc->cols;    // size of fourier descriptors
  assert(fdsize%2==0);                  // assume fdsize is even

  CvMat * matFourier =
  cvCreateMat(matContour->rows, matContour->cols, CV_32FC2);

  // perform fourier transform 
  cvDFT(matContour, matFourier, CV_DXT_FORWARD);

  // center - set first element to zero
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, 0) = cvPoint2D32f(0,0);

  // scale - divide by magnitude of 2nd element
  {
  CvPoint2D32f p =
  #if 1
  // last element
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, matFourier->cols-1); 
  #else
  // 2nd element
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, 1); 
  #endif
  float scale_coeff = sqrt((p.x*p.x)+(p.y*p.y));
  cvScale(matFourier, matFourier, 1.f/scale_coeff);
  }

  // crop - get correct size of fd
  int halflen = (conlen+1)/2-1;
  if (conlen<fdsize) // case of contour length less than fd size
  {
  for (int i = 0; i < halflen; i++){
  // front to middle
  CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, i) =
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, i+1);
  // back to middle
  CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, fdsize-1-i) =
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, conlen-1-i);
  }
  }else{
  for (int i = 0; i < fdsize/2; i++){
  // front to middle
  CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, i) =
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, i+1);
  // back to middle
  CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, fdsize-1-i) =
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, conlen-1-i);
  }
  }
  cvReleaseMatEx(matFourier);
  }
*/
