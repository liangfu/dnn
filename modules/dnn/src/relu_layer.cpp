/** -*- c++ -*- 
 *
 * \file   relu_layer.cpp
 * \date   Sat May 14 12:07:01 2016
 *
 * \copyright 
 * Copyright (c) 2016 Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the Brainnetome Center & NLPR at Institute of Automation, CAS. The 
 * name of the Brainnetome Center & NLPR at Institute of Automation, CAS 
 * may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * \brief  rectified linear unit
 */
 
#include "_dnn.h"

void cvReLU(CvMat * src, CvMat * dst) {cvMaxS( src, 0, dst );}

void cvReLUDer(CvMat * src, CvMat * dst) {
  CV_FUNCNAME("cvReLUDer");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_MAT_TYPE(dst->type));
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      if (srcptr[ii]>0){dstptr[ii] = 1.f;}else {dstptr[ii] = 0.f;}
    }
  }else if (CV_MAT_TYPE(src->type)==CV_64F){
    double * srcptr = src->data.db;
    double * dstptr = dst->data.db;
    for (ii=0;ii<elemsize;ii++){
      if (srcptr[ii]>0){dstptr[ii] = 1.f;}else {dstptr[ii] = 0.f;}
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}
