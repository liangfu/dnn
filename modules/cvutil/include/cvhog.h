/**
 * @file   cvhog.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Jun  4 17:53:32 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_HOG_H__
#define __CV_HOG_H__

#include "cvext_c.h"

void icvCalcHOG(CvMat * imgYpatch, CvMatND * hog,
                int ncells=-1, int ngrids=-1);
void icvCalcWarpHOG(CvMat * imgY, CvMat * warp_p, 
                    CvMatND * hog, int cellsize, CvSize bound,
                    CvMat * dx, CvMat * dy,
                    CvMat * magni_full, CvMat * angle_full);
void icvCalcWarpHOG(CvMat * imgY, CvMat * warp_p, 
                    CvMatND * hog, int ncells, int ngrids,
                    CvMat * dx, CvMat * dy,
                    CvMat * magni_full, CvMat * angle_full);
// void icvShowHOG(CvMatND * hog, int cmflag=CV_CM_HSV);
void icvShowHOG(CvMatND * hog, int cmflag=CV_CM_GRAY, int scale=1);

#endif // __CV_HOG_H__
