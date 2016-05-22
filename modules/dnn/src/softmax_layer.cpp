/** -*- c++ -*- 
 *
 * \file   softmax_layer.cpp
 * \date   Sat May 14 12:07:45 2016
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
 * \brief  softmax layer
 */

#include "_dnn.h"
 
//! assuming column vectors (a column is a sample)
void cvSoftmax(CvMat * src, CvMat * dst){
  CV_FUNCNAME("cvSoftmax");
  __BEGIN__;
  CV_ASSERT(cvCountNAN(src)<1);
  cvExp(src,dst);
  CV_ASSERT(cvCountNAN(dst)<1);
  const int dtype = CV_MAT_TYPE(src->type);
  CvMat * sum = cvCreateMat(1,src->cols,dtype);
  CvMat * sum_repeat = cvCreateMat(src->rows,src->cols,dtype);
  cvReduce(dst,sum,-1,CV_REDUCE_SUM);
  CV_ASSERT(cvCountNAN(sum)<1);
  cvRepeat(sum,sum_repeat);
  cvDiv(dst,sum_repeat,dst);
  cvReleaseMat(&sum);
  cvReleaseMat(&sum_repeat);
  __END__;
}

void cvSoftmaxDer(CvMat * X, CvMat * dE_dY, CvMat * dE_dY_afder) {
  CV_FUNCNAME("cvSoftmaxDer");
  __BEGIN__;
  const int nr = X->rows, nc = X->cols, dtype = CV_MAT_TYPE(X->type);
  CvMat * Y = cvCreateMat(nr, nc, dtype);
  CvMat * dE_dY_transpose = cvCreateMat(nr, nc, dtype);
  CvMat * sum = cvCreateMat(1, nc, dtype);
  CvMat * sum_repeat = cvCreateMat(nr, nc, dtype);
  cvSoftmax(X, Y);
  if (dE_dY->rows==nc && dE_dY->cols==nr){
    cvTranspose(dE_dY,dE_dY_transpose);
    cvMul(Y,dE_dY_transpose,dE_dY_afder);
  }else{
    cvMul(Y,dE_dY,dE_dY_afder);
  }
  cvReduce(dE_dY_afder,sum,-1,CV_REDUCE_SUM);
  cvRepeat(sum,sum_repeat);
  cvMul(Y,sum_repeat,sum_repeat);
  cvSub(dE_dY_afder,sum_repeat,dE_dY_afder);
  cvReleaseMat(&dE_dY_transpose);
  cvReleaseMat(&sum);
  cvReleaseMat(&sum_repeat);
  cvReleaseMat(&Y);
  __END__;
}
