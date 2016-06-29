/** -*- mode: c++ -*- 
 *
 * \file   _dnn.h
 * \date   Sat May 14 20:17:45 2016
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
 * \brief  internal functions for implementing layers
 */
#ifndef __DNN_H__
#define __DNN_H__

#include "cvext_c.h"
#include "layers.h"
#include "precomp.hpp"

// void cvCopyEx(CvMat * src, CvMat * dst);

CvMat * cvCloneTransposed(CvMat * src);

#define CV_GEMM(src1,src2,alpha,src3,beta,dst,tABC)                     \
  cvDebugGEMM(#src1,#src2,#src3,#dst,(src1),(src2),(alpha),(src3),(beta),(dst),(tABC));

void cvDebugGEMM(const char * src1name, const char * src2name, const char * src3name, const char * dstname,
                 CvMat * src1, CvMat * src2, float alpha, CvMat * src3, float beta, CvMat * dst, int tABC);
    
/*-------------- functions for input data layer -----------------------*/
void icvCNNInputRelease( CvDNNLayer** p_layer );
void icvCNNInputForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNInputBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );
    
/*-------------- functions for input data layer -----------------------*/
void icvCNNRepeatVectorRelease( CvDNNLayer** p_layer );
void icvCNNRepeatVectorForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNRepeatVectorBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*--------------- functions for convolutional layer --------------------*/
void icvCNNConvolutionRelease( CvDNNLayer** p_layer );
void icvCNNConvolutionForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNConvolutionBackward( CvDNNLayer*  layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX );

/*------------------ functions for sub-sampling layer -------------------*/
void icvCNNMaxPoolingRelease( CvDNNLayer** p_layer );
void icvCNNMaxPoolingForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNMaxPoolingBackward( CvDNNLayer*  layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX );

/*---------- functions for full connected layer -----------------------*/
void icvCNNDenseRelease( CvDNNLayer** p_layer );
void icvCNNDenseForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNDenseBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*-------------- functions for recurrent layer -----------------------*/
void icvCNNRecurrentRelease( CvDNNLayer** p_layer );
void icvCNNRecurrentForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNRecurrentBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*-------------- functions for multi target layer -----------------------*/
void icvCNNMergeRelease( CvDNNLayer** p_layer );
void icvCNNMergeForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNMergeBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

CvDNNLayer* icvCreateLayer( int layer_type, 
    const int dtype, const char * name, int header_size,
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int output_height, int output_width,
    float init_learn_rate, int learn_rate_decrease_type,
    CvDNNLayerRelease release, CvDNNLayerForward forward, CvDNNLayerBackward backward );

void icvVisualizeCNNLayer(CvDNNLayer * layer, const CvMat * Y);

#endif // __DNN_H__
