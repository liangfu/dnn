/** -*- c++ -*- 
 *
 * \file   subsample_layer.cpp
 * \date   Sat May 14 11:47:30 2016
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
 * \brief  create sub-sample of previous layer
 */
 
#include "_dnn.h"
#include "cvimgwarp.h"

/*-------------- functions for image cropping layer ------------------*/
static void icvCNNTimeDistributedRelease( CvDNNLayer** p_layer );
static void icvCNNTimeDistributedForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNTimeDistributedBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

CvDNNLayer * cvCreateTimeDistributedLayer( 
    const int dtype, const char * name, const int visualize, 
    const CvDNNLayer * _image_layer,
    int n_output_planes, int output_height, int output_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule
)
{
  CvDNNTimeDistributedLayer* layer = 0;
  int n_inputs = _image_layer->n_input_planes;
  int n_outputs = n_output_planes;
  CvDNNInputLayer * input_layer = (CvDNNInputLayer*)_image_layer;

  CV_FUNCNAME("cvCreateTimeDistributedLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }
  CV_ASSERT(icvIsInputLayer((CvDNNLayer*)_image_layer));

  fprintf(stderr,"TimeDistributedLayer(%s): "
          "input (%d@%dx%d), output (%d@%dx%d), seq_length: (%d), time_index: (%d)\n", name,
          n_inputs,input_layer->input_height,input_layer->input_width,
          n_outputs,output_height,output_width,input_layer->seq_length,time_index);
  
  CV_CALL(layer = (CvDNNTimeDistributedLayer*)icvCreateLayer( ICV_DNN_TIMEDISTRIBUTED_LAYER, dtype, name, 
      sizeof(CvDNNTimeDistributedLayer), 
      n_inputs, input_layer->input_height, input_layer->input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNTimeDistributedRelease, icvCNNTimeDistributedForward, icvCNNTimeDistributedBackward ));

  layer->input_layers.push_back((CvDNNLayer*)_image_layer);
  layer->seq_length = seq_length;
  layer->time_index = time_index;
  layer->visualize = visualize;

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvDNNLayer*)layer;
}

static void icvCNNTimeDistributedForward( CvDNNLayer * _layer, const CvMat * _X, CvMat * Y )
{
  CV_FUNCNAME("icvCNNTimeDistributedForward");
  if ( !icvIsTimeDistributedLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNTimeDistributedLayer * layer = (CvDNNTimeDistributedLayer*)_layer;
  const CvDNNInputLayer * input_layer = 
    (CvDNNInputLayer*)(layer->input_layers.size()>0?layer->input_layers[0]:layer->prev_layer);
  if (input_layer){
    if (!icvIsInputLayer((CvDNNLayer*)input_layer)){CV_ERROR(CV_StsBadArg,"invalid layer definition.");}
  }
  const int time_index = layer->time_index;
  const int input_seqlen = input_layer->seq_length;
  const int input_height = layer->input_height;
  const int input_width = layer->input_width;
  const int n_outputs = Y->cols;
  const int output_seqlen = layer->seq_length;
  const int output_height = layer->output_height;
  const int output_width = layer->output_width;
  const CvMat * X = (((CvDNNLayer*)input_layer)==layer->prev_layer)?_X:input_layer->Y;
  const int batch_size = icvIsInputLayer((CvDNNLayer*)input_layer)?(X->rows/input_layer->seq_length):X->rows;
  CV_ASSERT(Y->cols==layer->n_output_planes*layer->output_height*layer->output_width);
  CV_ASSERT(batch_size*input_seqlen==X->rows && batch_size*output_seqlen==Y->rows); // batch_size
  CvMat * input_data = input_layer->Y;
  CV_ASSERT(CV_MAT_TYPE(input_data->type)==CV_32F);
  if (input_seqlen>output_seqlen){ // temporal sampling
    CvMat input_data_submat;
    CvMat Y_submat_hdr;
    for (int bidx=0;bidx<batch_size;bidx++){
      // cvGetCol(input_data,&input_data_submat,bidx*input_seqlen+time_index);
      // cvGetCol(Y,&Y_submat_hdr,bidx);
      cvGetRow(input_data,&input_data_submat,bidx*input_seqlen+time_index);
      cvGetRow(Y,&Y_submat_hdr,bidx);
      cvCopy(&input_data_submat,&Y_submat_hdr);
    }
  }else{
    CV_ERROR(CV_StsBadArg,"invalid layer definition.");
  }
  if (layer->Y){cvCopy(Y,layer->Y);}else{layer->Y=cvCloneMat(Y);}
  if (layer->visualize){ icvVisualizeCNNLayer((CvDNNLayer*)layer, Y); }
  __END__;
}

static void icvCNNTimeDistributedBackward( CvDNNLayer* layer, int t, 
                                       const CvMat * X, const CvMat* dE_dY, CvMat* dE_dX )
{
}

static void icvCNNTimeDistributedRelease( CvDNNLayer** p_layer ){}
