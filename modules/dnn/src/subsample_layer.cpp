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
static void icvCNNImgCroppingRelease( CvCNNLayer** p_layer );
static void icvCNNImgCroppingForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNImgCroppingBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

CvCNNLayer * cvCreateCNNImgCroppingLayer( 
    const int dtype, const char * name, const int visualize, 
    const CvCNNLayer * _image_layer,
    int n_output_planes, int output_height, int output_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule
)
{
  CvCNNImgCroppingLayer* layer = 0;
  int n_inputs = _image_layer->n_input_planes;
  int n_outputs = n_output_planes;
  CvCNNInputDataLayer * input_layer = (CvCNNInputDataLayer*)_image_layer;

  CV_FUNCNAME("cvCreateCNNImgCroppingLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }
  CV_ASSERT(icvIsCNNInputDataLayer((CvCNNLayer*)_image_layer));

  fprintf(stderr,"ImgCroppingLayer(%s): "
          "input (%d@%dx%d), output (%d@%dx%d), seq_length: %d, time_index: %d\n", name,
          n_inputs,input_layer->input_height,input_layer->input_width,
          n_outputs,output_height,output_width,input_layer->seq_length,time_index);
  
  CV_CALL(layer = (CvCNNImgCroppingLayer*)icvCreateCNNLayer( ICV_CNN_IMGCROPPING_LAYER, dtype, name, 
      sizeof(CvCNNImgCroppingLayer), 
      n_inputs, input_layer->input_height, input_layer->input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNImgCroppingRelease, icvCNNImgCroppingForward, icvCNNImgCroppingBackward ));

  layer->input_layers.push_back((CvCNNLayer*)_image_layer);
  layer->seq_length = seq_length;
  layer->time_index = time_index;
  layer->visualize = visualize;

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}


static void icvCNNImgCroppingForward( CvCNNLayer * _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNImgCroppingForward");
  if ( !icvIsCNNImgCroppingLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvCNNImgCroppingLayer * layer = (CvCNNImgCroppingLayer*)_layer;
  const CvCNNInputDataLayer * input_layer = 
    (CvCNNInputDataLayer*)(layer->input_layers.size()>0?layer->input_layers[0]:0);
  const int time_index = layer->time_index;
  const int input_seqlen = input_layer->seq_length;
  const int input_height = layer->input_height;
  const int input_width = layer->input_width;
  const int n_outputs = Y->rows;
  const int output_seqlen = layer->seq_length;
  const int output_height = layer->output_height;
  const int output_width = layer->output_width;
  const int batch_size = X->cols/input_seqlen;
  CV_ASSERT(Y->rows==layer->n_output_planes*layer->output_height*layer->output_width);
  CV_ASSERT(batch_size*input_seqlen==X->cols && batch_size*output_seqlen==Y->cols); // batch_size
  CvMat * input_data = input_layer->input_data;
  CV_ASSERT(CV_MAT_TYPE(input_data->type)==CV_32F);
  if (input_seqlen>output_seqlen){ // temporal sampling
    CvMat input_data_submat;
    CvMat input_data_hdr = cvMat(input_seqlen*batch_size,n_outputs,CV_32F,input_data->data.ptr);
    cvGetRows(&input_data_hdr,&input_data_submat,
              batch_size*time_index,batch_size*time_index+batch_size);
    cvTranspose(&input_data_submat,Y);
  }else if (output_height==output_width && output_height>1 && output_width>1){ // image processing
    CvMat * I = input_data;
    CvMat * p = cvCreateMat(2,3,CV_32F); cvZero(p);
    if (X->rows==2){ CV_ASSERT(X->cols==1); // crop only
      CV_ASSERT(CV_MAT_ELEM(*X,float,0,0)>=-1.f && CV_MAT_ELEM(*X,float,0,0)<=1.f && 
                CV_MAT_ELEM(*X,float,1,0)>=-1.f && CV_MAT_ELEM(*X,float,1,0)<=1.f);
      float tx = float(input_width -output_width )*(CV_MAT_ELEM(*X,float,0,0)+1.f)*.5f;
      float ty = float(input_height-output_height)*(CV_MAT_ELEM(*X,float,1,0)+1.f)*.5f;
      CV_MAT_ELEM(*p,float,0,0)=CV_MAT_ELEM(*p,float,1,1)=1.f;
      CV_MAT_ELEM(*p,float,0,2)=tx;
      CV_MAT_ELEM(*p,float,1,2)=ty; // fprintf(stderr,"--\n"); cvPrintf(stderr,"%f ",p);
      CV_ASSERT(I->rows==input_height*input_width && CV_MAT_TYPE(I->type)==CV_32F);
      CV_ASSERT(Y->rows==output_height*output_width && CV_MAT_TYPE(Y->type)==CV_32F);
      CvMat srchdr = cvMat(input_height,input_width,CV_32F,I->data.ptr);
      CvMat dsthdr = cvMat(output_height,output_width,CV_32F,Y->data.ptr);
      icvWarp(&srchdr,&dsthdr,p); // CV_SHOW(&srchdr); CV_SHOW(&dsthdr);
    }else if (X->rows==3){ CV_ASSERT(X->cols==1); // crop and resize
    }else{ // resize image
      float scale = float(output_height)/float(input_height);
      CV_MAT_ELEM(*p,float,0,0)=1./scale;
      CV_MAT_ELEM(*p,float,1,1)=1./scale;
      CV_ASSERT(I->rows==input_height*input_width && CV_MAT_TYPE(I->type)==CV_32F);
      CV_ASSERT(Y->rows==output_height*output_width && CV_MAT_TYPE(Y->type)==CV_32F);
      CvMat srchdr = cvMat(input_height,input_width,CV_32F,I->data.ptr);
      CvMat dsthdr = cvMat(output_height,output_width,CV_32F,Y->data.ptr);
      icvWarp(&srchdr,&dsthdr,p); // CV_SHOW(&srchdr); CV_SHOW(&dsthdr);
    }
    cvReleaseMat(&p);
  }else{
    CV_Error(CV_StsBadArg,"invalid layer definition.");
  }
  if (layer->Y){cvCopy(Y,layer->Y);}else{layer->Y=cvCloneMat(Y);}
  if (layer->visualize){ icvVisualizeCNNLayer((CvCNNLayer*)layer, Y); }
  __END__;
}

static void icvCNNImgCroppingBackward( CvCNNLayer* layer, int t, 
                                       const CvMat * X, const CvMat* dE_dY, CvMat* dE_dX )
{
}

static void icvCNNImgCroppingRelease( CvCNNLayer** p_layer ){}
