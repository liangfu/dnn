/**
 * @file   cvimgwarp.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Mar 14 15:16:12 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_IMG_WARP_H__
#define __CV_IMG_WARP_H__

#include "cvext_c.h"

/** 
 * warp current frame to template size, using nearest-neighbor method
 * 
 * @param img       in:  input floating point image, assumed to be CV_32F
 * @param IWxp      out: output warped template image, assumed to be CV_32F
 * @param warp_p    in:  2x3 matrix defined as [R t],
 *                       represent for rotation and translation
 */
CVAPI(void) icvWarp(CvMat * img, CvMat * dst, CvMat * warp_p);
CVAPI(void) icvWarpInit(CvMat * warp_p, float tx, float ty);
CVAPI(void) icvWarpInvert(CvMat * warp_p, CvMat * invwarp_p3x3);
CVAPI(double) icvWarpToPoints(CvMat * warp_p,
                              CvPoint2D32f pts[4], int nr, int nc);
CVAPI(void) icvWarpCompose(CvMat * comp_M, CvMat * warp_p);
CVAPI(void) icvWarpTranspose(CvMat * delta_p, CvMat * delta_M);
CVAPI(void) icvWarpReshape(CvMat * warp_p, CvMat * warp_M);

#endif // __CV_IMG_WARP_H__
