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
static void icvCNNImgWarpingRelease( CvCNNLayer** p_layer );
static void icvCNNImgWarpingForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNImgWarpingBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

CvCNNLayer * cvCreateCNNImgWarpingLayer( 
    const int dtype, const char * name, const int visualize, 
    const CvCNNLayer * _image_layer,
    int n_output_planes, int output_height, int output_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule
)
{
  CvCNNImgWarpingLayer* layer = 0;
  int n_inputs = _image_layer->n_input_planes;
  int n_outputs = n_output_planes;
  CvCNNRepeatVectorLayer * input_layer = (CvCNNRepeatVectorLayer*)_image_layer;

  CV_FUNCNAME("cvCreateCNNImgWarpingLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }
  CV_ASSERT(icvIsCNNRepeatVectorLayer((CvCNNLayer*)_image_layer));

  fprintf(stderr,"ImgWarpingLayer(%s): "
          "input (%d@%dx%d), output (%d@%dx%d), seq_length: (%d), time_index: (%d)\n", name,
          n_inputs,input_layer->input_height,input_layer->input_width,
          n_outputs,output_height,output_width,input_layer->seq_length,time_index);
  
  CV_CALL(layer = (CvCNNImgWarpingLayer*)icvCreateCNNLayer( ICV_CNN_IMGWARPPING_LAYER, dtype, name, 
      sizeof(CvCNNImgWarpingLayer), 
      n_inputs, input_layer->input_height, input_layer->input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNImgWarpingRelease, icvCNNImgWarpingForward, icvCNNImgWarpingBackward ));

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


static void icvCNNImgWarpingForward( CvCNNLayer * _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNImgWarpingForward");
  if ( !icvIsCNNImgWarpingLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvCNNImgWarpingLayer * layer = (CvCNNImgWarpingLayer*)_layer;
  const CvCNNRepeatVectorLayer * input_layer = 
    (CvCNNRepeatVectorLayer*)(layer->input_layers.size()>0?layer->input_layers[0]:0);
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
  CvMat * input_data = input_layer->input_data;
  CV_ASSERT(CV_MAT_TYPE(input_data->type)==CV_32F);
  if (input_seqlen>output_seqlen){ // temporal sampling
    CvMat input_data_submat;
#if 0
    CvMat input_data_hdr = cvMat(input_seqlen*batch_size,n_outputs,CV_32F,input_data->data.ptr);
    cvGetRows(&input_data_hdr,&input_data_submat,
              batch_size*time_index,batch_size*time_index+batch_size);
    cvTranspose(&input_data_submat,Y);
#else
    CvMat Y_submat_hdr;
    for (int bidx=0;bidx<batch_size;bidx++){
      // cvGetCol(input_data,&input_data_submat,bidx*input_seqlen+time_index);
      // cvGetCol(Y,&Y_submat_hdr,bidx);
      cvGetRow(input_data,&input_data_submat,bidx*input_seqlen+time_index);
      cvGetRow(Y,&Y_submat_hdr,bidx);
      cvCopy(&input_data_submat,&Y_submat_hdr);
    }
#endif
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

static void icvCNNImgWarpingBackward( CvCNNLayer* layer, int t, 
                                       const CvMat * X, const CvMat* dE_dY, CvMat* dE_dX )
{
}

static void icvCNNImgWarpingRelease( CvCNNLayer** p_layer ){}
