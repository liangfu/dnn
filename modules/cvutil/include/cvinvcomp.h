/**
 * @file   cvinvcomp.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Sun Feb 17 11:59:13 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_INV_COMP_H__
#define __CV_INV_COMP_H__

#include "cvext_c.h"
#include "cvimgwarp.h"

/** 
 * calculate jacobian matrix for affine warp
 * 
 * @param dW_dp   out: 2x6 jacobian matrix for affine warp
 * @param nr      in:  number of rows of template image
 * @param nc      in:  number of cols of template image
 */
CVAPI(void) icvCalcJacobian(CvMat * dW_dp, const int nr, const int nc);

/** 
 * multiply jacobian with gradient images for gradient descent images
 * 
 * @param dW_dp     in:  the 2x6 template sized jacobian matrix 
 * @param nabla_Ix  in:  gradient image on X coordinate
 * @param nabla_Iy  in:  gradient image on Y coordinate
 * @param VI_dW_dp  out: 1x6 template sized gradient descent images,
 *                       assumed to be CV_32F
 * @param N_p       in:  number of tranformation parameters,
 *                       commonly defined to be 6
 */
CVAPI(void) icvCalcStDescImages(CvMat * dW_dp,
                                CvMat * nabla_Ix, CvMat * nabla_Iy,
                                CvMat * VI_dW_dp);

/** 
 * calculate 6x6 hessian matrix from 1x6 steepest descent images
 * 
 * @param VI_dW_dp in:  steepest descent images, assumed to be CV_32F
 * @param H        out: hessian matrix, assumed to be CV_64F
 */
CVAPI(void) icvCalcHessian(CvMat * VI_dW_dp, CvMat * H);

/** 
 * multiply sd_image with error_img to get sd parameters
 * 
 * @param VI_dW_dp    in:  steepest descent image,
 *                         the multiplication of gradient and jacobian
 * @param error_img   in:  error image = gradient - template
 * @param sd_delta_p  out: 6x1 matrix steepest descent 
 */
CVAPI(void) icvUpdateStDescImages(CvMat * VI_dW_dp, CvMat * error_img,
                                  CvMat * sd_delta_p);

#endif // __CV_INV_COMP_H__
