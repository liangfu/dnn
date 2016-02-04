/**
 * @file   cvhomography.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Aug 22 10:40:30 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_HOMOGRAPHY_H__
#define __CV_HOMOGRAPHY_H__

#include "cvext_c.h"

void icvGaussNewton( const CvMat* J, const CvMat* err, CvMat* delta,
                     CvMat* JtJ=0, CvMat* JtErr=0,
                     CvMat* JtJW=0, CvMat* JtJV=0 );
void icvConvertPointsHomogenious( const CvMat* src, CvMat* dst );
void icvFindHomography( const CvMat* object_points,
                        const CvMat* image_points, CvMat* __H );

#endif // __CV_HOMOGRAPHY_H__
