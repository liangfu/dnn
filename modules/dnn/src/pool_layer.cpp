/** -*- c++ -*- 
 *
 * \file   pool_layer.cpp
 * \date   Sat May 14 11:35:13 2016
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
 * \brief  max-pooling layer
 */

#include "_dnn.h" 

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNSubSamplingLayer( 
    const int dtype, const char * name, const int visualize,
    int n_input_planes, int input_height, int input_width,
    int sub_samp_scale, 
    float init_learn_rate, int learn_rate_decrease_type, CvMat* weights )

{
    CvCNNSubSamplingLayer* layer = 0;

    CV_FUNCNAME("cvCreateCNNSubSamplingLayer");
    __BEGIN__;

    const int output_height   = input_height/sub_samp_scale;
    const int output_width    = input_width/sub_samp_scale;
    const int n_output_planes = n_input_planes;
    fprintf(stderr,"SubSamplingLayer(%s): input (%d@%dx%d), output (%d@%dx%d)\n", name,
            n_input_planes,input_width,input_height,n_output_planes,output_width,output_height);

    if ( sub_samp_scale < 1 )
        CV_ERROR( CV_StsBadArg, "Incorrect parameters" );

    CV_CALL(layer = (CvCNNSubSamplingLayer*)icvCreateCNNLayer( 
        ICV_CNN_SUBSAMPLING_LAYER, dtype, name, sizeof(CvCNNSubSamplingLayer), 
        n_input_planes, input_height, input_width,
        n_output_planes, output_height, output_width,
        init_learn_rate, learn_rate_decrease_type,
        icvCNNSubSamplingRelease, icvCNNSubSamplingForward, icvCNNSubSamplingBackward ));

    layer->sub_samp_scale  = sub_samp_scale;
    layer->visualize = visualize;
    layer->seq_length = 1;
    layer->mask = 0;

    CV_CALL(layer->sumX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));
    CV_CALL(layer->WX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));
    
    cvZero( layer->sumX );
    cvZero( layer->WX );

    CV_CALL(layer->weights = cvCreateMat( n_output_planes, 2, CV_32FC1 ));
    if ( weights )
    {
        if ( !ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ) )
            CV_ERROR( CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1" );
        if ( !CV_ARE_SIZES_EQ( weights, layer->weights ) )
            CV_ERROR( CV_StsBadSize, "Invalid size of initial weights matrix" );
        CV_CALL(cvCopy( weights, layer->weights ));
    }
    else
    {
        CvRNG rng = cvRNG( 0xFFFFFFFF );
        cvRandArr( &rng, layer->weights, CV_RAND_UNI, cvRealScalar(-1), cvRealScalar(1) );
    }

    __END__;

    if ( cvGetErrStatus() < 0 && layer )
    {
        cvReleaseMat( &layer->WX );
        cvFree( &layer );
    }

    return (CvCNNLayer*)layer;
}

void icvCNNSubSamplingForward( CvCNNLayer* _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNSubSamplingForward");

  if ( !icvIsCNNSubSamplingLayer(_layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  __BEGIN__;

  CvCNNSubSamplingLayer * layer = (CvCNNSubSamplingLayer*) _layer;
  // CvMat * Xt = 0;
  // CvMat * Yt = 0;

  const int stride_size = layer->sub_samp_scale;
  // const int nsamples = X->rows; // batch size for training
  const int nplanes = layer->n_input_planes;
  const int Xheight = layer->input_height;
  const int Xwidth  = layer->input_width ;
  const int Xsize   = Xwidth*Xheight;
  const int Yheight = layer->output_height;
  const int Ywidth  = layer->output_width;
  const int Ysize   = Ywidth*Yheight;
  const int batch_size = X->rows;

  int xx, yy, ni, kx, ky;
  float* sumX_data = 0, *w = 0;
  CvMat sumX_sub_col, WX_sub_col;

  CV_ASSERT(X->cols == nplanes*Xsize && X->rows == batch_size);
  CV_ASSERT(Y->rows == batch_size);
  CV_ASSERT((layer->WX->cols == 1) && (layer->WX->rows == nplanes*Ysize));

  CV_CALL(layer->mask = cvCreateMat(batch_size, Ysize*nplanes, CV_32S));
  
  // update inner variable used in back-propagation
  cvZero( layer->sumX );
  cvZero( layer->WX );
  cvZero( layer->mask );
  
  int * mptr = layer->mask->data.i;

  CV_ASSERT(Xheight==Yheight*stride_size && Xwidth ==Ywidth *stride_size);
  CV_ASSERT(Y->rows==layer->mask->rows && Y->cols==layer->mask->cols);
  CV_ASSERT(CV_MAT_TYPE(layer->mask->type)==CV_32S);

  // Xt = cvCreateMat(X->cols,X->rows,CV_32F); cvTranspose(X,Xt);
  // Yt = cvCreateMat(Y->cols,Y->rows,CV_32F); cvTranspose(Y,Yt);
  for ( int si = 0; si < batch_size; si++ ){
  float * xptr = X->data.fl+Xsize*nplanes*si;
  float * yptr = Y->data.fl+Ysize*nplanes*si;
  for ( ni = 0; ni < nplanes; ni++ ){
    for ( yy = 0; yy < Yheight; yy++ ){
    for ( xx = 0; xx < Ywidth; xx++ ){
      float maxval = (xptr+stride_size*xx+Xwidth*0)[0];//*xptr;
      int maxloc = 0;
      for ( ky = 0; ky < stride_size; ky++ ){
      for ( kx = 0; kx < stride_size; kx++ ){
        if ((xptr+stride_size*xx+Xwidth*ky)[kx]>maxval) {
          maxval = (xptr+stride_size*xx+Xwidth*ky)[kx];
          maxloc = ky*stride_size + kx;
        }
      } // kx
      } // ky
      yptr[xx] = maxval;
      mptr[xx] = maxloc;
    } // xx
    xptr += Xwidth*stride_size;
    yptr += Ywidth;
    mptr += Ywidth;
    } // yy
  } // ni
  } // si
  // cvTranspose(Yt,Y);
  // cvReleaseMat(&Xt);
  // cvReleaseMat(&Yt);
  if (layer->Y){cvCopy(Y,layer->Y);}else{layer->Y=cvCloneMat(Y);}
  if (layer->visualize){icvVisualizeCNNLayer((CvCNNLayer*)layer,Y);}

  __END__;
}

void icvCNNSubSamplingBackward(
    CvCNNLayer* _layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX )
{
  // derivative of activation function
  CvMat* dY_dX_elems = 0; // elements of matrix dY_dX
  CvMat* dY_dW_elems = 0; // elements of matrix dY_dW
  CvMat* dE_dW = 0;

  CV_FUNCNAME("icvCNNSubSamplingBackward");

  if ( !icvIsCNNSubSamplingLayer(_layer) ) {
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  __BEGIN__;
  CvCNNSubSamplingLayer* layer = (CvCNNSubSamplingLayer*) _layer;

  const int Xwidth  = layer->input_width;
  const int Xheight = layer->input_height;
  const int Ywidth  = layer->output_width;
  const int Yheight = layer->output_height;
  const int Xsize   = Xwidth * Xheight;
  const int Ysize   = Ywidth * Yheight;
  const int scale   = layer->sub_samp_scale;
  const int k_max   = layer->n_output_planes * Yheight;

  int k, i, j, m;

  CV_ASSERT(CV_MAT_TYPE(layer->mask->type)==CV_32S);
  CV_ASSERT(dE_dX->rows*dE_dX->cols==X->rows*X->cols);

  int n_outputs = layer->n_output_planes;
  int batch_size = X->rows;
  int stride_size = layer->sub_samp_scale;
  CV_ASSERT(layer->mask->rows==batch_size && layer->mask->cols==n_outputs*Ysize);

  // CvMat * maskT = cvCreateMat(dE_dY->rows,dE_dY->cols,CV_32S);
  // cvTranspose(layer->mask,maskT);
  cvZero(dE_dX);
  for ( int si = 0; si < batch_size; si++ ){
  float * dxptr = dE_dX->data.fl+dE_dX->cols*si;
  float * dyptr = dE_dY->data.fl+dE_dY->cols*si;
  int * mptr = layer->mask->data.i;
  for ( int ni = 0; ni < n_outputs; ni++ ){
    for ( int yy = 0; yy < Yheight; yy++ ){
    for ( int xx = 0; xx < Ywidth; xx++ ){
      int maxloc = mptr[xx];
      int ky = maxloc / stride_size;
      int kx = maxloc % stride_size;
      (dxptr+stride_size*xx+Xwidth*ky)[kx]=dyptr[xx];
    }
    dxptr += Xwidth*stride_size;
    dyptr += Ywidth;
    mptr += Ywidth;
    }
  }
  }
  // cvReleaseMat(&maskT);
  
  if (layer->mask){cvReleaseMat(&layer->mask);layer->mask=0;}
  __END__;

  if (dY_dX_elems){cvReleaseMat( &dY_dX_elems );dY_dX_elems=0;}
  if (dY_dW_elems){cvReleaseMat( &dY_dW_elems );dY_dW_elems=0;}
  if (dE_dW      ){cvReleaseMat( &dE_dW       );dE_dW      =0;}
}

void icvCNNSubSamplingRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNSubSamplingRelease");
  __BEGIN__;

  CvCNNSubSamplingLayer* layer = 0;

  if ( !p_layer )
      CV_ERROR( CV_StsNullPtr, "Null double pointer" );

  layer = *(CvCNNSubSamplingLayer**)p_layer;

  if ( !layer )
      return;
  if ( !icvIsCNNSubSamplingLayer((CvCNNLayer*)layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  if (layer->mask      ){cvReleaseMat( &layer->mask       ); layer->mask      =0;}
  if (layer->WX){cvReleaseMat( &layer->WX ); layer->WX=0;}
  if (layer->weights   ){cvReleaseMat( &layer->weights    ); layer->weights   =0;}
  cvFree( p_layer );

  __END__;
}
