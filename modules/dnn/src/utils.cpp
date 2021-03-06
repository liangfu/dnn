/** -*- c++ -*- 
 *
 * \file   utils.cpp
 * \date   Sat May 21 17:47:52 2016
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
 * \brief  utility functions required in implement of dnn
 */
 
#include "_dnn.h"

void cvDebugGEMM(const char * src1name, const char * src2name, const char * src3name, const char * dstname,
                 CvMat * src1, CvMat * src2, float alpha, CvMat * src3, float beta, CvMat * dst, int tABC)
{
  if (((tABC)&CV_GEMM_A_T)==0 && ((tABC)&CV_GEMM_B_T)==0 &&          
      ((tABC)&CV_GEMM_C_T)==0){                                      
    if ((src1)->cols!=(src2)->rows){                                 
      fprintf(stderr,"Warning: %s(%dx%d), %s(%dx%d), %s(%dx%d)\n",   
              src1name,(src1)->rows,(src1)->cols,                       
              src2name,(src2)->rows,(src2)->cols,                       
              dstname,(dst)->rows,(dst)->cols);                         
    }                                                                
  }else if (((tABC)&CV_GEMM_A_T)>0 && ((tABC)&CV_GEMM_B_T)==0 &&     
            ((tABC)&CV_GEMM_C_T)==0){                                
    if ((src1)->rows!=(src2)->rows){                                 
      fprintf(stderr,"Warning: %s(%dx%d), %s(%dx%d), %s(%dx%d)\n",   
              src1name,(src1)->rows,(src1)->cols,                       
              src2name,(src2)->rows,(src2)->cols,                       
              dstname,(dst)->rows,(dst)->cols);                         
    }                                                                
  }else if (((tABC)&CV_GEMM_A_T)==0 && ((tABC)&CV_GEMM_B_T)>0 &&     
            ((tABC)&CV_GEMM_C_T)==0){                                
    if ((src1)->cols!=(src2)->cols){                                 
      fprintf(stderr,"Warning: %s(%dx%d), %s(%dx%d), %s(%dx%d)\n",   
              src1name,(src1)->rows,(src1)->cols,                       
              src2name,(src2)->rows,(src2)->cols,                       
              dstname,(dst)->rows,(dst)->cols);                         
    }                                                                
  }else if (((tABC)&CV_GEMM_A_T)>0 && ((tABC)&CV_GEMM_B_T)>0 &&      
            ((tABC)&CV_GEMM_C_T)==0){                                
    if ((src1)->cols!=(src2)->rows){                                 
      fprintf(stderr,"Warning: %s(%dx%d), %s(%dx%d), %s(%dx%d)\n",   
              src1name,(src1)->rows,(src1)->cols,                       
              src2name,(src2)->rows,(src2)->cols,                       
              dstname,(dst)->rows,(dst)->cols);                         
    }                                                                
  }                                                                  
  cvGEMM((src1),(src2),(alpha),(src3),(beta),(dst),(tABC));          
}

CvMat * cvCloneTransposed(CvMat * src)
{
  CvMat * dst = cvCreateMat(src->cols,src->rows,CV_MAT_TYPE(src->type));
  cvTranspose(src,dst);
  return dst;
}

void cvRandShuffleRows(CvMat * src, CvMat * _dst, CvRNG * rng)
{
  CV_FUNCNAME("cvRandShuffleRows");
  __BEGIN__;
  CvMat * dst=_dst;
  if (!_dst){dst=cvCloneMat(src);}
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_32F && CV_MAT_TYPE(dst->type)==CV_32F);
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  CV_ASSERT(src->step==src->cols*sizeof(float) && dst->step==dst->cols*sizeof(float));
  int n_samples = src->rows,k=0;
  CvMat * shuffle_idx = cvCreateMat(1,n_samples,CV_32S);
  for (int ii=0;ii<n_samples;ii++){CV_MAT_ELEM(*shuffle_idx,int,0,ii)=ii;}
  cvRandShuffle(shuffle_idx, rng, 1.f);
  for ( k = 0; k < n_samples; k++ ){
    memcpy(dst->data.fl+dst->cols*k,
           src->data.fl+dst->cols*shuffle_idx->data.i[k],
           sizeof(float)*dst->cols);
  }
  if (!_dst){cvCopy(dst,src);cvReleaseMat(&dst);}
  cvReleaseMat(&shuffle_idx);
  __END__;
}

void cvReorderRows(CvMat * src, CvMat * shuffle_idx)
{
  CV_FUNCNAME("cvRandShuffleRows");
  __BEGIN__;
  CvMat * dst=cvCloneMat(src);
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_32F && CV_MAT_TYPE(dst->type)==CV_32F);
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  CV_ASSERT(src->step==src->cols*sizeof(float) && dst->step==dst->cols*sizeof(float));
  int n_samples = src->rows,k=0;
  for ( k = 0; k < n_samples; k++ ){
    memcpy(dst->data.fl+dst->cols*k,
           src->data.fl+dst->cols*shuffle_idx->data.i[k],
           sizeof(float)*dst->cols);
  }
  cvCopy(dst,src);
  cvReleaseMat(&dst);
  __END__;
}


