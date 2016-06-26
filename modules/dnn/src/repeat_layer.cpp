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

CvDNNLayer * cvCreateRepeatVectorLayer( 
    const int dtype, const char * name, 
    int n_input_planes, int input_height, int input_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule)
{
  CvDNNRepeatVectorLayer* layer = 0;
  const int n_inputs = n_input_planes;
  const int n_outputs = n_input_planes;
  const int output_width = input_width;
  const int output_height = input_height;

  CV_FUNCNAME("cvCreateRepeatVectorLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  fprintf(stderr,"RepeatVectorLayer(%s): input (%d@%dx%d), output (%d@%dx%d), seq_length: (%d)\n", name,
          n_inputs,input_height,input_width,
          n_outputs,output_height,output_width,seq_length);
  
  CV_CALL(layer = (CvDNNRepeatVectorLayer*)icvCreateLayer( ICV_DNN_REPEATVECTOR_LAYER, dtype, name, 
      sizeof(CvDNNRepeatVectorLayer), n_inputs, input_height, input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNRepeatVectorRelease, icvCNNRepeatVectorForward, icvCNNRepeatVectorBackward ));

  layer->seq_length = seq_length;
  layer->time_index = 0;
  
  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvDNNLayer*)layer;
}

void icvCNNRepeatVectorForward( CvDNNLayer * _layer, const CvMat * X, CvMat * Y )
{
  CV_FUNCNAME("icvCNNRepeatVectorForward");
  if ( !icvIsRepeatVectorLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNRepeatVectorLayer * layer = (CvDNNRepeatVectorLayer*)_layer;
  if (!layer->Y){
    layer->Y = cvCloneMat(X);
  }else{
    if (X->cols==layer->Y->cols){
      cvCopy(X,layer->Y);
    }else{
      cvReleaseMat(&layer->Y);
      layer->Y = cvCloneMat(X);
    }
  }
  cvCopy(X,Y);
  __END__;
}

void icvCNNRepeatVectorBackward( CvDNNLayer * _layer, int t, 
                              const CvMat * X, const CvMat * _dE_dY, CvMat * dE_dX )
{
  CV_FUNCNAME("icvCNNRepeatVectorForward");
  if ( !icvIsRepeatVectorLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNRepeatVectorLayer * layer = (CvDNNRepeatVectorLayer*)_layer;
  CvDNNLayer * output_layer = layer->output_layers.size()>0?layer->output_layers[0]:0;
  const CvMat * dE_dY = (layer->output_layers.size()>0)?layer->output_layers[0]->dE_dX:_dE_dY;
  if (dE_dY->cols==dE_dX->cols){
    cvCopy(dE_dY,dE_dX);
  }else if (output_layer){
    if (icvIsMergeLayer(output_layer)){
      int n_input_layers = ((CvDNNMergeLayer*)output_layer)->input_layers.size();
      int layer_index = -1;
      int output_layer_index = 0;
      int output_layer_size = 0;
      for (int lidx=0;lidx<n_input_layers;lidx++){
        output_layer_index+=((CvDNNMergeLayer*)output_layer)->input_layers[lidx]->n_output_planes;
        if (!strcmp(((CvDNNMergeLayer*)output_layer)->input_layers[lidx]->name,layer->name)){
          layer_index=lidx;
          output_layer_index-=((CvDNNMergeLayer*)output_layer)->input_layers[lidx]->n_output_planes;
          output_layer_size=((CvDNNMergeLayer*)output_layer)->input_layers[lidx]->n_output_planes;
          break;
        }
      }
      if (layer_index>=0){
        const int batch_size = dE_dY->rows;
        CvMat * dE_dY_submat = cvCreateMat(batch_size,output_layer_size,CV_32F);
        CvMat dE_dX_submat_hdr; 
        cvGetCols(output_layer->dE_dX,&dE_dX_submat_hdr,output_layer_index,output_layer_index+output_layer_size);
        cvCopy(&dE_dX_submat_hdr,dE_dY_submat);
        cvCopy(dE_dY_submat,dE_dX);
        cvReleaseMat(&dE_dY_submat);
      }else{CV_ERROR(CV_StsBadArg, "output_layer->input_layer should be current layer.");}
    }
  }else{
    CV_ERROR(CV_StsBadArg,"invalid layer definition.");
  }
  __END__;
}

void icvCNNRepeatVectorRelease( CvDNNLayer** p_layer ){}

