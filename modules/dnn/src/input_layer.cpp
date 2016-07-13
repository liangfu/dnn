/** -*- c++ -*- 
 *
 * \file   input_layer.cpp
 * \date   Sat May 14 11:55:59 2016
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
 * \brief  contain original raw input data
 */
 
#include "_dnn.h"

CvDNNLayer * cvCreateInputLayer( 
    const int dtype, const char * name, 
    int n_input_planes, int input_height, int input_width, int seq_length,
    float init_learn_rate, int update_rule)
{
  CvDNNInputLayer* layer = 0;
  const int n_inputs = n_input_planes;
  const int n_outputs = n_input_planes;
  const int output_width = input_width;
  const int output_height = input_height;

  CV_FUNCNAME("cvCreateInputLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  fprintf(stderr,"InputLayer(%s): input (%d@%dx%d), output (%d@%dx%d), seq_length: (%d)\n", name,
          n_inputs,input_height,input_width,
          n_outputs,output_height,output_width,seq_length);
  
  CV_CALL(layer = (CvDNNInputLayer*)icvCreateLayer( ICV_DNN_INPUTDATA_LAYER, dtype, name, 
      sizeof(CvDNNInputLayer), n_inputs, input_height, input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNInputRelease, icvCNNInputForward, icvCNNInputBackward ));

  layer->seq_length = seq_length;

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvDNNLayer*)layer;
}

void icvCNNInputForward( CvDNNLayer * _layer, const CvMat * X, CvMat * Y )
{
  CV_FUNCNAME("icvCNNInputForward");
  if ( !icvIsInputLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNInputLayer * layer = (CvDNNInputLayer*)_layer;
  CV_ASSERT(cvCountNAN((CvMat*)X)<1);
  if (!layer->Y){
    layer->Y = cvCloneMat(X);
  }else{
    if (X->cols==layer->Y->cols && X->rows==layer->Y->rows){
      cvCopy(X,layer->Y);
    }else{
      cvReleaseMat(&layer->Y);
      layer->Y = cvCloneMat(X);
    }
  }
  cvCopy(X,Y);
  CV_ASSERT(cvCountNAN(Y)<1);
  if (layer->visualize){icvVisualizeCNNLayer((CvDNNLayer*)layer,Y);}
  __END__;
}

void icvCNNInputBackward( CvDNNLayer * layer, int t, 
                              const CvMat * X, const CvMat * dE_dY, CvMat * dE_dX )
{
  cvZero(dE_dX);
}

void icvCNNInputRelease( CvDNNLayer** p_layer ){}

