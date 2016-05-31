/** -*- c++ -*- 
 *
 * \file   combine_layer.cpp
 * \date   Sat May 14 12:00:11 2016
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
 * \brief  combine multiple input layer results
 */
 
#include "_dnn.h"

CvCNNLayer * cvCreateCNNMultiTargetLayer( 
    const int dtype, const char * name, const int visualize,
    int n_input_layers, CvCNNLayer ** input_layers, int n_outputs,
    float init_learn_rate, int update_rule)
{
  CvCNNMultiTargetLayer * layer = 0;
  int n_inputs = n_input_layers;
  int input_width = 1, input_height = 1, output_width = 1, output_height = 1;

  CV_FUNCNAME("cvCreateCNNMultiTargetLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  fprintf(stderr,"MultiTargetLayer(%s): input_layers (%d), outputs (%d)\n", name,
          n_input_layers,n_outputs);
  
  CV_CALL(layer = (CvCNNMultiTargetLayer*)icvCreateCNNLayer( ICV_CNN_MULTITARGET_LAYER, dtype, name, 
      sizeof(CvCNNMultiTargetLayer), n_inputs, input_height, input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNMultiTargetRelease, icvCNNMultiTargetForward, icvCNNMultiTargetBackward ));

  layer->seq_length = 1;
  layer->visualize = visualize;
  // layer->n_input_layers = n_input_layers;
  // layer->input_layers = new CvCNNLayer*[n_input_layers];
  for (int lidx=0; lidx<n_input_layers; lidx++){
    // layer->input_layers[lidx] = input_layers[lidx];
    layer->input_layers.push_back(input_layers[lidx]);
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

void icvCNNMultiTargetForward( CvCNNLayer * _layer, const CvMat * X, CvMat * Y )
{
  CV_FUNCNAME("icvCNNMultiTargetForward");
  __BEGIN__;
  CvCNNMultiTargetLayer * layer = (CvCNNMultiTargetLayer*)_layer;
  int n_input_layers = layer->input_layers.size();
  // CvCNNLayer ** input_layers = layer->input_layers;
  int input_layer_data_index = 0;
  for (int lidx=0;lidx<n_input_layers;lidx++){
    CvMat Y_submat;
    int input_layer_data_size = layer->input_layers[lidx]->n_output_planes;
    cvGetCols(Y,&Y_submat,input_layer_data_index,input_layer_data_index+input_layer_data_size);
    cvCopy(layer->input_layers[lidx]->Y,&Y_submat);
    input_layer_data_index += input_layer_data_size;
  }
  if (layer->visualize){ icvVisualizeCNNLayer((CvCNNLayer*)layer, Y); }
  __END__;
}

void icvCNNMultiTargetBackward( CvCNNLayer* layer, int t,
                                       const CvMat*, const CvMat* dE_dY, CvMat* dE_dX )
{
  CV_FUNCNAME("icvCNNMultiTargetBackward");
  __BEGIN__;
  if (layer->dE_dX){cvCopy(dE_dY,layer->dE_dX);}else{layer->dE_dX=cvCloneMat(dE_dY);}
  __END__;
}

void icvCNNMultiTargetRelease( CvCNNLayer** p_layer ){}
