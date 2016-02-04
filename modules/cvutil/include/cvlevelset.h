/**
 * @file   cvlevelset.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Dec 11 14:06:05 2012
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_EXT_LEVEL_SET_H__
#define __CV_EXT_LEVEL_SET_H__

#include "cvext_c.h"

CVAPI(void) icvInitializeLevelSet(CvMat * mask, CvMat * phi);

/** 
 * get sum of region from integral image
 * 
 * @param _integral in: integral image
 * @param roi       in: rectangle of ROI
 * @param sum       out: result sum
 */
CV_INLINE
void cvGetRegionSum(CvArr * _integral, CvRect roi, int & sum)
{
	CvMat header; CvMat * integral = cvGetMat(_integral, &header);
	assert(CV_MAT_TYPE(integral->type)==CV_32S);
	int lt = CV_MAT_ELEM(*integral, int, roi.y, roi.x);
	int rt = CV_MAT_ELEM(*integral, int, roi.y, roi.x+roi.width);
	int rb = CV_MAT_ELEM(*integral, int, roi.y+roi.height, roi.x+roi.width);
	int lb = CV_MAT_ELEM(*integral, int, roi.y+roi.height, roi.x);
	sum = rb-(rt+lb-lt); 
}

CV_INLINE
void cvGetRegionSum(CvArr * _integral, CvRect roi, double & sum)
{
	CvMat header; CvMat * integral = cvGetMat(_integral, &header);
	assert(CV_MAT_TYPE(integral->type)==CV_64F);
	double lt = CV_MAT_ELEM(*integral, double, roi.y, roi.x);
	double rt = CV_MAT_ELEM(*integral, double, roi.y, roi.x+roi.width);
	double rb = CV_MAT_ELEM(*integral, double, roi.y+roi.height, roi.x+roi.width);
	double lb = CV_MAT_ELEM(*integral, double, roi.y+roi.height, roi.x);
	sum = rb-(rt+lb-lt); 
}

//--------------------------------
// fast sweep non-zero region within given ROI and the seed, output
// collected points into queue sequence.
// Example:
//   CvMemStorage * storage = cvCreateMemStorage();
//   // .. generate contour here ..
//   CvSeq * seq = cvCreateSeq(
//     CV_32SC2,            // sequence of integer elements
// 	   sizeof(CvSeq),       // header size - no extra fields
// 	   sizeof(CvPoint),     // element size
// 	   storage /* the container storage */ );
//   CvPoint seed = cvPoint(12,12);
//   cvSweep(binary_image, cvRect(10,10,100,100), seed, seq);
//   ...
// 
// imgSrc : source image
// roi : region of interest
// pt : seed point
// qSeq : queue, as list of sequence points
// return value: area of the propagated region
CVAPI(void) cvSweep(CvArr * src, CvRect roi, CvPoint pt, 
             CvSeq * qSeq ); // queue

/** 
 * dilate binary image with seed
 * 
 * @param _img 
 * @param seed 
 * @param dist 
 */
CVAPI(void) cvSeededBinaryDilate(CvArr * _img, CvPoint seed,
                                 const int dist CV_DEFAULT(5));

CVAPI(void) cvCalcDivergence(CvMat * dx, CvMat * dy, CvMat * dst);
CVAPI(void) cvCalcDirac(CvMat * src, CvMat * dst, const float sigma=1.2);
CVAPI(void) cvCalcHeaviside(CvMat * src, CvMat * dst, const float _eps=1.2);
CVAPI(void) cvNeumannBoundCond(CvMat * src);

struct CvLevelSetTracker_
{
  int type;           // internally used floating-type:
                      // either CV_32F or CV_64F
  float lambda;       // weighted length term
  float mu;           // weighted distance term
  float alpha;        // weighted area term
  float epsilon;      // width of Dirac Delta function
  float dt;           // time step
  int inner_maxiter;      // number of iterations
  int outer_maxiter;      // number of iterations

  CvSize imsize;
  CvMat * phi;
  CvMat * g;

  CvMat * bw;
  CvMat * vx;
  CvMat * vy;
  CvMat * dx;
  CvMat * dy;
  CvMat * mag;
  CvMat * Nx;
  CvMat * Ny;
  CvMat * dxdx;
  CvMat * dydy;
  CvMat * curvature;
  CvMat * del2;
  CvMat * dirac;

  CvMat * distRegTerm;
  CvMat * edgeTerm;
  CvMat * areaTerm;
};

typedef struct CvLevelSetTracker_ CvLevelSetTracker;

CVAPI(CvLevelSetTracker*) cvCreateLevelSetTracker(
    const CvSize imsize,
    const int type CV_DEFAULT(CV_32F));

CVAPI(void) cvLevelSetPrepare(CvLevelSetTracker * lstracker,
                              const CvArr * img,
                              CvRect * rois, int nroi,
                              int ksize);

CVAPI(int) cvLevelSetUpdate(CvLevelSetTracker * t,
                            float _dt,
                            float _mu, float _lambda, float _alpha, 
                            // float dt, float alpha, 
                            int inner_maxiter,
                            int outer_maxiter);

CVAPI(void) cvReleaseLevelSetTracker(CvLevelSetTracker ** lstracker);

#endif // __CV_EXT_LEVEL_SET_H__
