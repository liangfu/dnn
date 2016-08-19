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
void icvCNNSpatialTransformRelease( CvDNNLayer** p_layer );
void icvCNNSpatialTransformForward( CvDNNLayer* layer, const CvMat* X, CvMat* Y );
void icvCNNSpatialTransformBackward( CvDNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

CvDNNLayer * cvCreateSpatialTransformLayer( 
    const int dtype, const char * name, const int visualize, 
    const CvDNNLayer * _image_layer,
    int n_output_planes, int output_height, int output_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule
)
{
  CvDNNSpatialTransformLayer* layer = 0;
  int n_inputs = _image_layer->n_input_planes;
  int n_outputs = n_output_planes;
  CvDNNInputLayer * input_layer = (CvDNNInputLayer*)_image_layer;

  CV_FUNCNAME("cvCreateSpatialTransformLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }
  CV_ASSERT(icvIsInputLayer((CvDNNLayer*)_image_layer));

  fprintf(stderr,"SpatialTransformLayer(%s): "
          "input (%d@%dx%d), output (%d@%dx%d)\n", name,
          n_inputs,input_layer->input_height,input_layer->input_width,
          n_outputs,output_height,output_width);
  
  CV_CALL(layer = (CvDNNSpatialTransformLayer*)icvCreateLayer( ICV_DNN_IMGWARPPING_LAYER, dtype, name, 
      sizeof(CvDNNSpatialTransformLayer), 
      n_inputs, input_layer->input_height, input_layer->input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNSpatialTransformRelease, icvCNNSpatialTransformForward, icvCNNSpatialTransformBackward ));

  char layername_fc1[20]; sprintf(layername_fc1,"%s_fc1",name);
  char layername_fc2[20]; sprintf(layername_fc2,"%s_fc2",name);
  const int n_hiddens = exp((log(n_inputs)+log(n_outputs))*.5f);
  layer->fc1_layer = cvCreateDenseLayer(dtype,layername_fc1,0,_image_layer,
    n_inputs,n_hiddens,init_learn_rate,CV_DNN_LEARN_RATE_DECREASE_SQRT_INV,"tanh",0);
  layer->fc2_layer = cvCreateDenseLayer(dtype,layername_fc1,0,0,
    n_hiddens,n_outputs,init_learn_rate,CV_DNN_LEARN_RATE_DECREASE_SQRT_INV,"none",0);
  layer->G = 0; // sampling grid -- initalized in forward pass
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


void icvCNNSpatialTransformForward( CvDNNLayer * _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNSpatialTransformForward");
  if ( !icvIsSpatialTransformLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNSpatialTransformLayer * layer = (CvDNNSpatialTransformLayer*)_layer;
  const CvDNNInputLayer * input_layer = 
    (CvDNNInputLayer*)(layer->input_layers.size()>0?layer->input_layers[0]:0);
  const int time_index = layer->time_index;
  const int input_seqlen = input_layer->seq_length;
  const int input_height = layer->input_height;
  const int input_width = layer->input_width;
  const int n_outputs = Y->cols;
  const int output_seqlen = layer->seq_length;
  const int output_height = layer->output_height;
  const int output_width = layer->output_width;
  const int batch_size = X->rows/input_seqlen;
  CV_ASSERT(Y->cols==layer->n_output_planes*layer->output_height*layer->output_width);
  CV_ASSERT(batch_size*input_seqlen==X->rows && batch_size*output_seqlen==Y->rows); // batch_size
  CvMat * input_data = input_layer->Y;
  CV_ASSERT(CV_MAT_TYPE(input_data->type)==CV_32F);
  CV_ASSERT(output_height==output_width && output_height>1 && output_width>1);

  // icvCNNDenseForward(_layer,X,Y);
  
  { // image processing
    CvMat * I = input_data;
    CvMat * p = cvCreateMat(2,3,CV_32F); cvZero(p);
    if (X->cols==2){ CV_ASSERT(X->rows==1); // translation only
      CV_ASSERT(CV_MAT_ELEM(*X,float,0,0)>=0.f && CV_MAT_ELEM(*X,float,0,0)<=1.f && 
                CV_MAT_ELEM(*X,float,0,1)>=0.f && CV_MAT_ELEM(*X,float,0,1)<=1.f);
      float tx = float(input_width -output_width )*CV_MAT_ELEM(*X,float,0,0);
      float ty = float(input_height-output_height)*CV_MAT_ELEM(*X,float,0,1);
      CV_MAT_ELEM(*p,float,0,0)=CV_MAT_ELEM(*p,float,1,1)=1.f;
      CV_MAT_ELEM(*p,float,0,2)=tx;
      CV_MAT_ELEM(*p,float,1,2)=ty; // fprintf(stderr,"--\n"); cvPrintf(stderr,"%f ",p);
      int nchannels=I->cols/(input_height*input_width);
      CV_ASSERT(I->cols==input_height*input_width*nchannels && CV_MAT_TYPE(I->type)==CV_32F);
      CV_ASSERT(Y->cols==output_height*output_width*nchannels && CV_MAT_TYPE(Y->type)==CV_32F);
      for (int chidx=0; chidx<nchannels; chidx++){
        CvMat srchdr = cvMat(input_height,input_width,CV_32F,I->data.fl+input_height*input_width*chidx);
        CvMat dsthdr = cvMat(output_height,output_width,CV_32F,Y->data.fl+output_height*output_width*chidx);
        icvWarp(&srchdr,&dsthdr,p);
      }
      // CV_SHOW(&srchdr); CV_SHOW(&dsthdr);
    }else if (X->cols==3){ CV_ASSERT(X->rows==1); // crop and resize
    }else{ // resize image
      float scale = float(output_height)/float(input_height);
      CV_MAT_ELEM(*p,float,0,0)=1./scale;
      CV_MAT_ELEM(*p,float,1,1)=1./scale;
      CV_ASSERT(I->cols==input_height*input_width && CV_MAT_TYPE(I->type)==CV_32F);
      CV_ASSERT(Y->cols==output_height*output_width && CV_MAT_TYPE(Y->type)==CV_32F);
      CvMat srchdr = cvMat(input_height,input_width,CV_32F,I->data.ptr);
      CvMat dsthdr = cvMat(output_height,output_width,CV_32F,Y->data.ptr);
      icvWarp(&srchdr,&dsthdr,p); // CV_SHOW(&srchdr); CV_SHOW(&dsthdr);
    }
    cvReleaseMat(&p);
  }
  if (layer->Y){cvCopy(Y,layer->Y);}else{layer->Y=cvCloneMat(Y);}
  if (layer->visualize){ icvVisualizeCNNLayer((CvDNNLayer*)layer, Y); }
  __END__;
}

void icvCNNSpatialTransformBackward( CvDNNLayer* _layer, int t, 
                                       const CvMat * X, const CvMat* dE_dY, CvMat* dE_dX )
{
  CV_FUNCNAME("icvCNNSpatialTransformBackward");
  if ( !icvIsSpatialTransformLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvDNNSpatialTransformLayer * layer = (CvDNNSpatialTransformLayer*)_layer;
  const CvDNNInputLayer * input_layer = 
    (CvDNNInputLayer*)(layer->input_layers.size()>0?layer->input_layers[0]:0);
  const CvMat * I = input_layer->Y;
  // CV_ASSERT(I->rows==dE_dX->rows && I->cols==dE_dX->cols);
  const int time_index = layer->time_index;
  const int input_seqlen = input_layer->seq_length;
  const int input_height = layer->input_height;
  const int input_width = layer->input_width;
  const int n_outputs = dE_dY->cols;
  const int output_seqlen = layer->seq_length;
  const int output_height = layer->output_height;
  const int output_width = layer->output_width;
  const int batch_size = X->rows/input_seqlen;
  CV_ASSERT(dE_dY->cols==layer->n_output_planes*layer->output_height*layer->output_width);
  CV_ASSERT(batch_size*input_seqlen==X->rows && batch_size*output_seqlen==dE_dY->rows);
  CV_ASSERT(CV_MAT_TYPE(I->type)==CV_32F);
  CV_ASSERT(output_height==output_width && output_height>1 && output_width>1);
  CvMat * p = cvCreateMat(2,3,CV_32F); cvZero(p);
  CvMat * M = cvCreateMat(3,3,CV_32F); cvZero(M);
  CvMat * invM = cvCreateMat(3,3,CV_32F); cvZero(invM);
  if (X->cols==2){ CV_ASSERT(X->rows==1); // translation only
    CV_ASSERT(CV_MAT_ELEM(*X,float,0,0)>=0.f && CV_MAT_ELEM(*X,float,0,0)<=1.f && 
              CV_MAT_ELEM(*X,float,0,1)>=0.f && CV_MAT_ELEM(*X,float,0,1)<=1.f);
    float tx=float(input_width -output_width )*CV_MAT_ELEM(*X,float,0,0);
    float ty=float(input_height-output_height)*CV_MAT_ELEM(*X,float,0,1);
    CV_MAT_ELEM(*M,float,0,0)=CV_MAT_ELEM(*M,float,1,1)=CV_MAT_ELEM(*M,float,2,2)=1.f;
    CV_MAT_ELEM(*M,float,0,2)=tx; CV_MAT_ELEM(*M,float,1,2)=ty;
    cvInvert(M,invM);
    memcpy(p->data.fl,M->data.fl,sizeof(float)*6); // cvPrintf(stderr,"%f ",p);
    // CvMat srchdr = cvMat(input_height,input_width,CV_32F,I->data.ptr);
    // CvMat dsthdr = cvMat(output_height,output_width,CV_32F,Y->data.ptr);
    // icvWarp(&srchdr,&dsthdr,p); // CV_SHOW(&srchdr); CV_SHOW(&dsthdr);
  }else{CV_ERROR(CV_StsBadArg,"invalid layer definition.");}
  cvReleaseMat(&M);
  cvReleaseMat(&invM);
  cvReleaseMat(&p);
  __END__;
}

void icvCNNSpatialTransformRelease( CvDNNLayer** p_layer ){}
