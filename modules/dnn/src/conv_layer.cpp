/** -*- c++ -*- 
 *
 * \file   conv_layer.cpp
 * \date   Sat May 14 11:30:53 2016
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
 * \brief  Convolutional Neural Network (ConvNet,CNN) layer
 */

#include "_dnn.h"

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNConvolutionLayer( 
    const int dtype, const char * name, const CvCNNLayer * ref_layer,
    const int visualize, const CvCNNLayer * input_layer, 
    int n_input_planes, int input_height, int input_width, int n_output_planes, int K,
    float init_learn_rate, int learn_rate_decrease_type,
    CvMat* connect_mask, CvMat* weights )

{
  CvCNNConvolutionLayer* layer = 0;

  CV_FUNCNAME("cvCreateCNNConvolutionLayer");
  __BEGIN__;

  const int output_height = input_height - K + 1;
  const int output_width = input_width - K + 1;
  fprintf(stderr,"ConvolutionLayer(%s): input (%d@%dx%d), output (%d@%dx%d)\n", name,
          n_input_planes,input_width,input_height,
          n_output_planes,output_width,output_height);

  if ( K < 1 || init_learn_rate <= 0 || init_learn_rate > 1 ) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  CV_CALL(layer = (CvCNNConvolutionLayer*)icvCreateCNNLayer( 
    ICV_CNN_CONVOLUTION_LAYER, dtype, name, sizeof(CvCNNConvolutionLayer), 
    n_input_planes, input_height, input_width,
    n_output_planes, output_height, output_width,
    init_learn_rate, learn_rate_decrease_type,
    icvCNNConvolutionRelease, icvCNNConvolutionForward, icvCNNConvolutionBackward ));

  layer->enable_cache = 1;
  layer->K = K;
  layer->seq_length = 1;
  layer->visualize = visualize;
  layer->ref_layer = (CvCNNLayer*)ref_layer;
  if (input_layer){layer->input_layers.push_back((CvCNNLayer*)input_layer);}
  CV_CALL(layer->weights = cvCreateMat( n_output_planes, K*K+1, CV_32FC1 ));
  CV_CALL(layer->connect_mask = cvCreateMat( n_output_planes, n_input_planes, CV_8UC1));

  if ( weights ){
    if ( !ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ) ) {
      CV_ERROR( CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1" );
    }
    if ( !CV_ARE_SIZES_EQ( weights, layer->weights ) ) {
      CV_ERROR( CV_StsBadSize, "Invalid size of initial weights matrix" );
    }
    CV_CALL(cvCopy( weights, layer->weights ));
  }else{
    CvRNG rng = cvRNG( -1 ); float invKK = 1./float(K*K);
    cvRandArr( &rng, layer->weights, CV_RAND_UNI, cvScalar(-1), cvScalar(1) );
    // normalize weights
    CvMat * sum = cvCreateMat(n_output_planes,1,CV_32F);
    CvMat * sumrep = cvCreateMat(n_output_planes,layer->weights->cols,CV_32F);
    cvReduce(layer->weights,sum,-1,CV_REDUCE_SUM); cvScale(sum,sum,invKK);
    cvRepeat(sum,sumrep);
    cvSub(layer->weights,sumrep,layer->weights);
    cvReleaseMat(&sum);
    cvReleaseMat(&sumrep);
    // initialize bias to zero
    for (int ii=0;ii<layer->weights->rows;ii++){ CV_MAT_ELEM(*layer->weights,float,ii,K*K)=0; }
  }

  if ( connect_mask ) {
    if ( !ICV_IS_MAT_OF_TYPE( connect_mask, CV_8UC1 ) ) {
      CV_ERROR( CV_StsBadSize, "Type of connection matrix must be CV_32FC1" );
    }
    if ( !CV_ARE_SIZES_EQ( connect_mask, layer->connect_mask ) ) {
      CV_ERROR( CV_StsBadSize, "Invalid size of connection matrix" );
    }
    CV_CALL(cvCopy( connect_mask, layer->connect_mask ));
  }else{
    CV_CALL(cvSet( layer->connect_mask, cvRealScalar(1) ));
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvReleaseMat( &layer->weights );
    cvReleaseMat( &layer->connect_mask );
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

void icvCNNConvolutionForward( CvCNNLayer* _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNConvolutionForward");

  if (!icvIsCNNConvolutionLayer(_layer)){CV_ERROR( CV_StsBadArg, "Invalid layer" );}

  __BEGIN__;

  CvCNNConvolutionLayer* layer = (CvCNNConvolutionLayer*) _layer;
  CvCNNLayer * ref_layer = layer->ref_layer;
  CvMat * weights = ref_layer?ref_layer->weights:layer->weights;
  
  const int K = layer->K;
  const int n_weights_for_Yplane = K*K + 1;
  CV_ASSERT(weights->cols==n_weights_for_Yplane);

  const int nXplanes = layer->n_input_planes;
  const int Xheight  = layer->input_height;
  const int Xwidth   = layer->input_width ;
  const int Xsize    = Xwidth*Xheight;

  const int nYplanes = layer->n_output_planes;
  const int Yheight  = layer->output_height;
  const int Ywidth   = layer->output_width;
  const int Ysize    = Ywidth*Yheight;

  const int nsamples = X->cols; // training batch size

  // int no; xx, yy, ni, , kx, ky
  // float *Yplane = 0, *Xplane = 0, *w = 0;
  uchar* connect_mask_data = 0;

  CV_ASSERT( X->rows == nXplanes*Xsize && X->cols == nsamples );
  CV_ASSERT( Y->rows == nYplanes*Ysize && Y->cols == nsamples );
  CV_ASSERT( Xheight-K+1 == Yheight && Xwidth-K+1 == Ywidth );

  cvSetZero( Y );

  // Yplane = Y->data.fl;
  // w = layer->weights->data.fl;
  connect_mask_data = layer->connect_mask->data.ptr;
  CvMat * Xt = cvCreateMat(X->cols,X->rows,CV_32F); cvTranspose(X,Xt);
  CvMat * Yt = cvCreateMat(Y->cols,Y->rows,CV_32F); cvTranspose(Y,Yt);

  // normalize input
  CvScalar avg,sdv;
  for ( int si = 0; si < nsamples; si++ ){
  for ( int no = 0; no < nXplanes; no++ ){
    float * xptr = Xt->data.fl+Xsize*nXplanes*si+Xsize*no;
    CvMat img = cvMat(Xsize,1,CV_32F,xptr);
    cvAvgSdv(&img,&avg,&sdv);
    cvSubS(&img,avg,&img);
    cvScale(&img,&img,.5f/sdv.val[0]);
  }
  }
  
  // for ( no = 0; no < nYplanes; no++, Yplane += Ysize, w += n_weights_for_Yplane ){
#pragma omp parallel for
  for ( int si = 0; si < nsamples; si++ ){
    for ( int no = 0; no < nYplanes; no++ ){
    float * xptr = Xt->data.fl+Xsize*nXplanes*si;
    float * yptr = Yt->data.fl+Ysize*nYplanes*si+Ysize*no;
    float * wptr = weights->data.fl+n_weights_for_Yplane*no;
    for ( int ni = 0; ni < nXplanes; ni++, xptr += Xsize ){
      for ( int yy = 0; yy < Xheight-K+1; yy++ ){
      for ( int xx = 0; xx < Xwidth-K+1; xx++ ){
        float WX = 0;
        for ( int ky = 0; ky < K; ky++ ){
        for ( int kx = 0; kx < K; kx++ ){
          WX += (xptr+Xwidth*yy+xx)[Xwidth*ky+kx]*wptr[K*ky+kx];
        } // kx
        } // ky
        yptr[(Xwidth-K+1)*yy+xx] += WX + wptr[K*K];
      } // xx
      } // yy
    } // ni
    } // no
  } // si

  cvScale(Yt,Yt,1.f/float(K*K));
  
  cvTranspose(Yt,Y); 
  cvReleaseMat(&Xt);
  cvReleaseMat(&Yt);
  
  if (layer->Y){cvCopy(Y,layer->Y);}else{layer->Y=cvCloneMat(Y);}
  if (layer->visualize){icvVisualizeCNNLayer((CvCNNLayer*)layer,Y);}

  __END__;
}

/* <dE_dY>, <dE_dX> should be row-vectors.
   Function computes partial derivatives <dE_dX>
   of the loss function with respect to the planes components
   of the previous layer (X).
   It is a basic function for back propagation method.
   Input parameter <dE_dY> is the partial derivative of the
   loss function with respect to the planes components
   of the current layer. */
void icvCNNConvolutionBackward(
    CvCNNLayer * _layer, int t, const CvMat* X, const CvMat* _dE_dY, CvMat* dE_dX )
{
  CV_FUNCNAME("icvCNNConvolutionBackward");
  if ( !icvIsCNNConvolutionLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }

  __BEGIN__;

  CvCNNConvolutionLayer * layer = (CvCNNConvolutionLayer*) _layer;
  int n_output_layers = layer->output_layers.size();
  CvCNNLayer * ref_layer = layer->ref_layer;
  CvMat * weights = ref_layer?ref_layer->weights:layer->weights;
  
  const int K = layer->K;
  const int KK = K*K;

  const int n_X_planes     = layer->n_input_planes;
  const int Xheight = layer->input_height;
  const int Xwidth  = layer->input_width;
  const int X_plane_size   = Xheight*Xwidth;

  const int n_Y_planes     = layer->n_output_planes;
  const int Yheight = layer->output_height;
  const int Ywidth  = layer->output_width;
  const int Y_plane_size   = Yheight*Ywidth;

  const int batch_size = X->cols;
  CvMat * dE_dY = (CvMat*)_dE_dY;
  CvMat* dY_dX = 0;
  CvMat* dY_dW = 0;
  CvMat* dE_dW = 0;

  if (n_output_layers){
    dE_dY = cvCreateMat(batch_size,Y_plane_size*n_Y_planes,CV_32F); cvZero(dE_dY);
    for (int li=0;li<n_output_layers;li++){
      CvCNNLayer * output_layer = layer->output_layers[li];
      if (icvIsCNNFullConnectLayer(output_layer)){
        cvAddWeighted(dE_dY,1.f,output_layer->dE_dX,1.f/float(n_output_layers),0.f,dE_dY);
      }
    } // average loss from all task
  }

  CV_ASSERT( t >= 1 );
  CV_ASSERT( n_Y_planes == weights->rows );

  if (layer->enable_cache){
    if (!layer->dY_dX){layer->dY_dX=cvCreateMat( n_Y_planes*Y_plane_size, X->rows, CV_32F );}
    dY_dX = layer->dY_dX;
  }else{
    dY_dX = cvCreateMat( n_Y_planes*Y_plane_size, X->rows, CV_32F );
  }
  dY_dW = cvCreateMat( dY_dX->rows, weights->cols*weights->rows, CV_32F );
  dE_dW = cvCreateMat( 1, dY_dW->cols, CV_32F );

  cvZero( dY_dX );
  cvZero( dY_dW );

  // compute gradient of the loss function with respect to X and W
  CvMat * Xt = cvCreateMat(X->cols,X->rows,CV_32F); cvTranspose(X,Xt);
#pragma omp parallel for
  for ( int si = 0; si < batch_size; si++ ){
    int yloc = 0;
    for ( int no = 0; no < n_Y_planes; no++, yloc += Y_plane_size ){
    int noKK = no*(KK+1);
    int xloc = 0;
    float * xptr = Xt->data.fl+Xt->cols*si;
    float * wptr = weights->data.fl + noKK;
    for ( int ni = 0; ni < n_X_planes; ni++, xptr += X_plane_size, xloc += X_plane_size ){
      for ( int yy = 0; yy < Xheight - K + 1; yy++ ){
      for ( int xx = 0; xx < Xwidth - K + 1; xx++ ){
        for ( int ky = 0; ky < K; ky++ ){
        for ( int kx = 0; kx < K; kx++ ){
          int kidx = K*ky+kx; // weights
          int ridx = Ywidth*yy+xx;
          int cidx = Xwidth*(yy+ky)+(xx+kx);
          CV_MAT_ELEM(*dY_dX,float,yloc+ridx,xloc+cidx) = wptr[kidx];
          CV_MAT_ELEM(*dY_dW,float,yloc+ridx,noKK+kidx) += xptr[cidx];
        } // ky
        } // kx
        int ridx = Ywidth*yy+xx;
        CV_MAT_ELEM(*dY_dW, float, yloc+ridx, noKK+KK) += 1; // bias
      } // xx
      } // yy
    } // ni
    } // no
  } // si
  cvReleaseMat(&Xt);
  cvScale(dY_dW,dY_dW,1.f/float(batch_size));

  // dE_dW = sum( dE_dY * dY_dW )
  CvMat * dE_dW_ = cvCreateMat( batch_size, dY_dW->cols, CV_32FC1 );
  CV_CALL(cvMatMul( dE_dY, dY_dW, dE_dW_ )); 
  cvReduce(dE_dW_,dE_dW,-1,CV_REDUCE_AVG);
  cvReleaseMat(&dE_dW_);

  // dE_dX = dE_dY * dY_dX
  CV_CALL(cvMatMul( dE_dY, dY_dX, dE_dX ));

  // update weights
  {
    CvMat dE_dW_mat;
    float eta;
    if ( layer->decay_type == CV_CNN_LEARN_RATE_DECREASE_LOG_INV ) {
      eta = -layer->init_learn_rate/logf(1+(float)t);
    } else if ( layer->decay_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV ) {
      eta = -layer->init_learn_rate/sqrtf((float)t);
    } else {
      eta = -layer->init_learn_rate/(float)t;
    }
    cvReshape( dE_dW, &dE_dW_mat, 0, weights->rows );
    if (!layer->dE_dW){
      ((CvCNNLayer*)layer)->dE_dW = cvCloneMat(&dE_dW_mat);
    }else{
      cvCopy(&dE_dW_mat,((CvCNNLayer*)layer)->dE_dW);
    }
    cvScaleAdd( &dE_dW_mat, cvRealScalar(eta), weights, weights );
  }

  if (n_output_layers){cvReleaseMat(&dE_dY);dE_dY=0;}
  if (!layer->enable_cache){
    if (dY_dX){cvReleaseMat( &dY_dX );dY_dX=0;}
  }
  if (dY_dW){cvReleaseMat( &dY_dW );dY_dW=0;}
  if (dE_dW){cvReleaseMat( &dE_dW );dE_dW=0;}

  __END__;
}

void icvCNNConvolutionRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNConvolutionRelease");
  __BEGIN__;

  CvCNNConvolutionLayer* layer = 0;

  if ( !p_layer )
      CV_ERROR( CV_StsNullPtr, "Null double pointer" );

  layer = *(CvCNNConvolutionLayer**)p_layer;

  if ( !layer )
      return;
  if ( !icvIsCNNConvolutionLayer((CvCNNLayer*)layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  if (layer->weights){cvReleaseMat( &layer->weights );layer->weights=0;}
  cvReleaseMat( &layer->connect_mask );
  cvFree( p_layer );

  __END__;
}

