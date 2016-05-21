/** -*- c++ -*- 
 *
 * \file   rnn_layer.cpp
 * \date   Sat May 14 11:43:20 2016
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
 * \brief  recurrent neural network (RNN) layer
 */
 
#include "_dnn.h"

ML_IMPL CvCNNLayer* cvCreateCNNRecurrentLayer( 
    const int dtype, const char * name, const CvCNNLayer * ref_layer, 
    int n_inputs, int n_outputs, int n_hiddens, int seq_length, int time_index, 
    float init_learn_rate, int update_rule, const char * activation_type, 
    CvMat * Wxh, CvMat * Whh, CvMat * Why )
{
  CvCNNRecurrentLayer* layer = 0;

  CV_FUNCNAME("cvCreateCNNRecurrentLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }

  fprintf(stderr,"RecurrentNNLayer(%s): input(%d), hidden(%d), output(%d), "
          "seq_length(%d), time_index(%d)\n", name,
          n_inputs, n_hiddens, n_outputs, seq_length, time_index);
  
  CV_CALL(layer = (CvCNNRecurrentLayer*)icvCreateCNNLayer( ICV_CNN_RECURRENTNN_LAYER, dtype, name, 
      sizeof(CvCNNRecurrentLayer), n_inputs, 1, 1, n_outputs, 1, 1,
      init_learn_rate, update_rule,
      icvCNNRecurrentRelease, icvCNNRecurrentForward, icvCNNRecurrentBackward ));

  layer->ref_layer = (CvCNNLayer*)ref_layer;
  layer->weights = 0; // we don't use this !
  layer->time_index = time_index;
  layer->seq_length = seq_length;
  layer->n_hiddens = n_hiddens;
  layer->Wxh = 0;
  layer->Whh = 0;
  layer->Why = 0;
  strcpy(layer->activation_type,activation_type);
  layer->H = 0;
  layer->Y = 0;
  layer->loss = 0;
  layer->dE_dY = 0;

  int n_hiddens = layer->n_hiddens;
  if (!ref_layer){
    CV_CALL(layer->Wxh = cvCreateMat( n_hiddens, n_inputs , CV_32F ));
    CV_CALL(layer->Whh = cvCreateMat( n_hiddens, n_hiddens+1, CV_32F ));
    CV_CALL(layer->Why = cvCreateMat( n_outputs, n_hiddens+1, CV_32F ));
    if ( Wxh || Whh || Why  ){
      CV_CALL(cvCopy( Wxh, layer->Wxh ));
      CV_CALL(cvCopy( Whh, layer->Whh ));
      CV_CALL(cvCopy( Why, layer->Why ));
    } else {
      CvRNG rng = cvRNG( -1 );
      cvRandArr( &rng, layer->Wxh, CV_RAND_UNI, 
                 cvScalar(-1.f/sqrt(n_inputs)), cvScalar(1.f/sqrt(n_inputs)) );
      cvRandArr( &rng, layer->Whh, CV_RAND_UNI, 
                 cvScalar(-1.f/sqrt(n_hiddens)), cvScalar(1.f/sqrt(n_hiddens)) );
      cvRandArr( &rng, layer->Why, CV_RAND_UNI, 
                 cvScalar(-1.f/sqrt(n_hiddens)), cvScalar(1.f/sqrt(n_hiddens)) );
      for (int ii=0;ii<n_hiddens;ii++){ CV_MAT_ELEM(*layer->Whh,float,ii,n_hiddens)=0; }
      for (int ii=0;ii<n_outputs;ii++){ CV_MAT_ELEM(*layer->Why,float,ii,n_hiddens)=0; }
    }
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvReleaseMat( &layer->Wxh );
    cvReleaseMat( &layer->Whh );
    cvReleaseMat( &layer->Why );
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

void icvCNNRecurrentRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNRecurrentRelease");
  __BEGIN__;

  CvCNNRecurrentLayer* layer = 0;

  if ( !p_layer ) { CV_ERROR( CV_StsNullPtr, "Null double pointer" ); }

  layer = *(CvCNNRecurrentLayer**)p_layer;

  if ( !layer ) { return; }
  if ( !icvIsCNNRecurrentNNLayer((CvCNNLayer*)layer) ) { 
    CV_ERROR( CV_StsBadArg, "Invalid layer" ); 
  }

  cvReleaseMat( &layer->Wxh ); layer->Wxh = 0;
  cvReleaseMat( &layer->Whh ); layer->Whh = 0;
  cvReleaseMat( &layer->Why ); layer->Why = 0;
  if (layer->dE_dY){cvReleaseMat(&layer->dE_dY);layer->dE_dY=0;}
  cvFree( p_layer );

  __END__;
}

/****************************************************************************************/
void icvCNNRecurrentForward( CvCNNLayer* _layer, const CvMat* X, CvMat * Y) 
{
  CV_FUNCNAME("icvCNNRecurrentForward");
  if ( !icvIsCNNRecurrentNNLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;

  CvCNNRecurrentLayer * layer = (CvCNNRecurrentLayer*)_layer;
  CvCNNRecurrentLayer * ref_layer = (CvCNNRecurrentLayer*)layer->ref_layer;
  CvMat Wxh_submat, Whh_submat, hbiascol, Why_submat, ybiascol;
  int time_index = layer->time_index;
  int seq_length = layer->seq_length;
  int n_inputs = layer->n_input_planes;//Y->rows;
  int n_outputs = layer->n_output_planes;//Y->rows;
  int n_hiddens = layer->n_hiddens;
  int batch_size = X->cols;
  CvMat * WX = 0, * WH = 0, * H_prev = 0, * H_curr = 0, * WX_curr, * WH_curr;

  CV_ASSERT(X->cols == batch_size && X->rows == layer->n_input_planes);

  // memory allocation
  if (!ref_layer){
    CV_ASSERT(!layer->H && !layer->Y && !layer->WX && !layer->WH);
    layer->H = cvCreateMat( n_hiddens * batch_size, seq_length, CV_32F ); cvZero(layer->H);
    layer->Y = cvCreateMat( n_outputs * batch_size, seq_length, CV_32F ); cvZero(layer->Y);
    layer->WX = cvCreateMat( n_hiddens * batch_size, seq_length, CV_32F ); cvZero(layer->WX);
    layer->WH = cvCreateMat( n_outputs * batch_size, seq_length, CV_32F ); cvZero(layer->WH);
  }
  CvMat * Wxh = ref_layer?ref_layer->Wxh:layer->Wxh;
  CvMat * Whh = ref_layer?ref_layer->Whh:layer->Whh;
  CvMat * Why = ref_layer?ref_layer->Why:layer->Why;
  CvMat * layerH = ref_layer?ref_layer->H:layer->H;
  CvMat * layerY = ref_layer?ref_layer->Y:layer->Y;
  CvMat * layerWX = ref_layer?ref_layer->WX:layer->WX;
  CvMat * layerWH = ref_layer?ref_layer->WH:layer->WH;
  CV_ASSERT(cvGetSize(layerH)==cvGetSize(layerWX));
  CV_CALL(WX = cvCreateMat( n_hiddens, batch_size, CV_32F ));
  CV_CALL(WH = cvCreateMat( n_hiddens, batch_size, CV_32F ));
  CV_CALL(H_prev = cvCreateMat( n_hiddens * batch_size, 1, CV_32F ));
  CV_CALL(H_curr = cvCreateMat( n_hiddens * batch_size, 1, CV_32F ));
  CV_CALL(WX_curr = cvCreateMat( n_hiddens * batch_size, 1, CV_32F ));
  CV_CALL(WH_curr = cvCreateMat( n_outputs * batch_size, 1, CV_32F ));
  cvZero( WX ); cvZero( WH );
  
  // bias on last column vector
  CV_CALL(cvGetCols( Whh, &Whh_submat, 0, Whh->cols-1));
  CV_CALL(cvGetCols( Why, &Why_submat, 0, Why->cols-1));
  CV_CALL(cvGetCol( Whh, &hbiascol, Whh->cols-1));
  CV_CALL(cvGetCol( Why, &ybiascol, Why->cols-1));
  CvMat * hbias = cvCreateMat(hbiascol.rows,batch_size,CV_32F);
  CvMat * ybias = cvCreateMat(ybiascol.rows,batch_size,CV_32F);
  cvRepeat(&hbiascol,hbias);
  cvRepeat(&ybiascol,ybias);

  // hidden states
  CvMat H_prev_hdr, H_curr_hdr, WX_curr_hdr, WH_curr_hdr, Y_curr_hdr;
  if (layer->time_index==0){ cvZero(H_prev); }else{
    cvGetCol(layerH,&H_prev_hdr,layer->time_index-1); cvCopy(&H_prev_hdr,H_prev);
  }
  cvGetCol(layerWX,&WX_curr_hdr,layer->time_index); cvCopy(&WX_curr_hdr,WX_curr);
  cvGetCol(layerWH,&WH_curr_hdr,layer->time_index); cvCopy(&WH_curr_hdr,WH_curr);
  CvMat H_prev_reshaped = cvMat(n_hiddens, batch_size, CV_32F, H_prev->data.ptr);
  CvMat H_curr_reshaped = cvMat(n_hiddens, batch_size, CV_32F, H_curr->data.ptr);
  CvMat WX_curr_reshaped = cvMat(n_hiddens, batch_size, CV_32F, WX_curr->data.ptr);
  CvMat WH_curr_reshaped = cvMat(n_outputs, batch_size, CV_32F, WH_curr->data.ptr);

  // H_t1 = Wxh * X_t0 + ( Whh * H_t0 + bh )
  CV_CALL(cvGEMM( Wxh, X, 1, 0, 1, WX ));
  CV_CALL(cvGEMM( &Whh_submat, &H_prev_reshaped, 1, hbias, 1, WH ));
  cvAdd(WX,WH,&H_curr_reshaped);
  cvCopy(&H_curr_reshaped,&WX_curr_reshaped);
  
  // activation for hidden states, relu and tahn is preferred
  cvTanh( H_curr, H_curr ); 

  // CV_ASSERT(batch_size==1);
  cvGetCol(layerH,&H_curr_hdr,layer->time_index); cvCopy(H_curr,&H_curr_hdr);
  cvGetCol(layerY,&Y_curr_hdr,layer->time_index);
  CvMat * Y_curr = cvCloneMat(&Y_curr_hdr), Y_curr_reshape_hdr;
  cvReshape(Y_curr,&Y_curr_reshape_hdr,0,n_outputs);
  CV_ASSERT(Y_curr_reshape_hdr.rows==n_outputs && Y_curr_reshape_hdr.cols==batch_size);

  // Y = sigmoid(Why * H + by)
  // CV_CALL(cvGEMM( &Why_submat, &H_curr_reshaped, 1, ybias, 1, &Y_curr_hdr ));
  CV_CALL(cvGEMM( &Why_submat, &H_curr_reshaped, 1, ybias, 1, &Y_curr_reshape_hdr ));
  CV_CALL(cvCopy(&Y_curr_reshape_hdr,&WH_curr_reshaped)); CV_ASSERT(cvCountNonZero(&WH_curr_reshaped)>1);
  CV_CALL(cvCopy(WH_curr,&WH_curr_hdr));          // copy to layer->WH
#if 0
  CV_CALL(cvSigmoid( &Y_curr_hdr, &Y_curr_hdr )); // output activation - logistic regression
#else
  cvSoftmax(&Y_curr_reshape_hdr,&Y_curr_reshape_hdr);
  // cvExp(&Y_curr_reshape_hdr,&Y_curr_reshape_hdr);
  // double Ysum = cvSum(&Y_curr_reshape_hdr).val[0];
  // cvScale(&Y_curr_hdr,&Y_curr_hdr,1.f/Ysum);      // softmax - for classification
#endif
  cvCopy(Y_curr,&Y_curr_hdr);

#if 0
  cvCopy(&Y_curr_hdr,Y);
#else
  // if (layer->time_index!=layer->seq_length-1) { cvCopy(&Y_curr_hdr,Y); }else
  {
    // int nr = (ref_layer?ref_layer:layer)->Y->rows;
    // int nc = (ref_layer?ref_layer:layer)->Y->cols;
    // CvMat * ref_layer_Y_transpose = cvCreateMat(nc,nr,CV_32F);
    // cvTranspose((ref_layer?ref_layer:layer)->Y,ref_layer_Y_transpose);
    // CvMat ref_layer_Y_reshaped = cvMat(nr*nc,1,CV_32F,ref_layer_Y_transpose->data.ptr);
    // cvCopy(&ref_layer_Y_reshaped,Y); 
    // cvReleaseMat(&ref_layer_Y_transpose);
    CvMat layerY_transpose_reshape_hdr;
    CvMat * layerY_transpose = cvCreateMat(layerY->cols,layerY->rows,CV_32F); 
    CvMat * Y_transpose = cvCreateMat(Y->cols,Y->rows,CV_32F);
    cvZero(layerY_transpose); cvZero(Y_transpose);
    cvTranspose(layerY,layerY_transpose);
    CV_ASSERT(Y->rows==n_outputs && Y->cols==batch_size*seq_length);
    cvReshape(layerY_transpose,&layerY_transpose_reshape_hdr,0,Y->cols);
    cvCopy(&layerY_transpose_reshape_hdr,Y_transpose);
    cvTranspose(Y_transpose,Y);
    cvReleaseMat(&layerY_transpose);
    cvReleaseMat(&Y_transpose);
  }
#endif
  if (Y_curr){cvReleaseMat(&Y_curr);Y_curr=0;}
 
  if (WX){cvReleaseMat(&WX);WX=0;}
  if (WH){cvReleaseMat(&WH);WH=0;}
  if (H_prev){cvReleaseMat(&H_prev);H_prev=0;}
  if (H_curr){cvReleaseMat(&H_curr);H_curr=0;}
  if (WX_curr){cvReleaseMat(&WX_curr);WX_curr=0;}
  if (WH_curr){cvReleaseMat(&WH_curr);WH_curr=0;}
  if (hbias){cvReleaseMat(&hbias);hbias=0;}
  if (ybias){cvReleaseMat(&ybias);ybias=0;}

  __END__;
}


/****************************************************************************************/
/* <dE_dY>, <dE_dX> should be row-vectors.
   Function computes partial derivatives <dE_dX>, <dE_dW>
   of the loss function with respect to the planes components
   of the previous layer (X) and the weights of the current layer (W)
   and updates weights od the current layer by using <dE_dW>.
   It is a basic function for back propagation method.
   Input parameter <dE_dY> is the partial derivative of the
   loss function with respect to the planes components
   of the current layer. */
void icvCNNRecurrentBackward( CvCNNLayer* _layer, int t,
                                     const CvMat * X, const CvMat * _dE_dY, CvMat * dE_dX )
{
  CV_FUNCNAME( "icvCNNRecurrentBackward" );
  if ( !icvIsCNNRecurrentNNLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }

  __BEGIN__;

  CvCNNRecurrentLayer * layer = (CvCNNRecurrentLayer*)_layer;
  CvCNNRecurrentLayer * ref_layer = (CvCNNRecurrentLayer*)layer->ref_layer;
  CvMat * dE_dY = (CvMat*)_dE_dY;
  
  // TODO: compute average from all output_layers
  int n_output_layers = ref_layer?ref_layer->output_layers.size():layer->output_layers.size();
  if (n_output_layers){
    const int n_Y_planes = layer->n_output_planes;
    const int Y_plane_size   = layer->output_height*layer->output_width;
    const int batch_size = X->cols;
    dE_dY = cvCreateMat(batch_size,Y_plane_size*n_Y_planes,CV_32F); cvZero(dE_dY);
    for (int li=0;li<n_output_layers;li++){
      CvCNNLayer * output_layer = ref_layer?ref_layer->output_layers[li]:layer->output_layers[li];
      if (icvIsCNNFullConnectLayer(output_layer)){
        cvAddWeighted(dE_dY,1.f,output_layer->dE_dX,1.f/float(n_output_layers),0.f,dE_dY);
      }
    }
  }

  CvMat * layer_Wxh = ref_layer?ref_layer->Wxh:layer->Wxh;
  CvMat * layer_Whh = ref_layer?ref_layer->Whh:layer->Whh;
  CvMat * layer_Why = ref_layer?ref_layer->Why:layer->Why;
  CvMat * layerH = ref_layer?ref_layer->H:layer->H;
  CvMat * layerY = ref_layer?ref_layer->Y:layer->Y;
  CvMat * layer_WX = ref_layer?ref_layer->WX:layer->WX;
  CvMat * layer_WH = ref_layer?ref_layer->WH:layer->WH;
  CvMat * layer_dE_dY = ref_layer?ref_layer->dE_dY:layer->dE_dY;
  CvMat * layer_dH = ref_layer?ref_layer->dH:layer->dH;
  CvMat * layer_dWxh = ref_layer?ref_layer->dWxh:layer->dWxh;
  CvMat * layer_dWhh = ref_layer?ref_layer->dWhh:layer->dWhh;
  CvMat * layer_dWhy = ref_layer?ref_layer->dWhy:layer->dWhy;
  CvMat  Whh_submat,  hbiascol,  Why_submat,  ybiascol;
  CvMat dWhh_submat, dhbiascol, dWhy_submat, dybiascol;
  CvMat layer_dWhh_submat, layer_dhbiascol, layer_dWhy_submat, layer_dybiascol;
  int time_index = layer->time_index;
  int seq_length = layer->seq_length;
  int n_inputs = layer->n_input_planes;
  int n_outputs = layer->n_output_planes;
  int n_hiddens = layer->n_hiddens;
  int batch_size = X->cols;
  CvMat * dE_dY_afder = 0;
  CvMat * WX = 0, * WH = 0, * H_prev = 0, * H_curr = 0, * WX_curr = 0, * WH_curr = 0;
  CvMat * dE_dY_curr = 0, * dH_curr = 0, * dH_next = 0, * dH_raw = 0, 
        * dWxh = 0, * dWhh = 0, * dWhy = 0;

  CV_ASSERT( cvGetSize(layerH)==cvGetSize(layer_WX) );
  if ( !ref_layer ){ CV_ASSERT(layer->H && layer->Y && layer->WX && layer->WH); }
  if ( ref_layer ){
    if ( layer->time_index==seq_length-1 ){  // assuming last layer
      if (!ref_layer->dE_dY){
        CV_ASSERT(dE_dY->cols==n_outputs);
        ref_layer->dE_dY = cvCloneMat(dE_dY); layer_dE_dY = ref_layer->dE_dY;
      }else{ cvCopy(dE_dY,ref_layer->dE_dY); }
      if (!ref_layer->dH){ 
        CV_ASSERT(!ref_layer->dWxh && !ref_layer->dWhh && !ref_layer->dWhy);
        ref_layer->dH   = cvCreateMat(layerH->rows,  layerH->cols,  CV_32F);
        ref_layer->dWxh = cvCreateMat(layer_Wxh->rows,layer_Wxh->cols,CV_32F);
        ref_layer->dWhh = cvCreateMat(layer_Whh->rows,layer_Whh->cols,CV_32F);
        ref_layer->dWhy = cvCreateMat(layer_Why->rows,layer_Why->cols,CV_32F);
      }
      cvZero(ref_layer->dH  ); layer_dH  =ref_layer->dH  ;
      cvZero(ref_layer->dWxh); layer_dWxh=ref_layer->dWxh;
      cvZero(ref_layer->dWhh); layer_dWhh=ref_layer->dWhh;
      cvZero(ref_layer->dWhy); layer_dWhy=ref_layer->dWhy;
    }else{ 
      CV_ASSERT(dE_dY->cols==n_outputs && layer_dE_dY==ref_layer->dE_dY && 
                layer_dH==ref_layer->dH && layer_dWxh==ref_layer->dWxh && 
                layer_dWhh==ref_layer->dWhh   && layer_dWhy==ref_layer->dWhy);
      CV_ASSERT(layer_dH && layer_dWxh && layer_dWhh && layer_dWhy);
    }
  }else{
    CV_ASSERT(dE_dY->cols==n_outputs && layer_dE_dY==layer->dE_dY && 
              layer_dH==layer->dH && layer_dWxh==layer->dWxh &&
              layer_dWhh==layer->dWhh && layer_dWhy==layer->dWhy); 
    CV_ASSERT(layer_dE_dY && layer_dH && layer_dWxh && layer_dWhh && layer_dWhy);
    {CvScalar avg, sdv; cvAvgSdv(layer_dE_dY,&avg,&sdv); CV_ASSERT(sdv.val[0]>1e-5);}
  }
  
  // memory allocation
  CV_CALL(WX = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero( WX );
  CV_CALL(WH = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero( WH );
  CV_CALL(H_prev = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(H_prev);
  CV_CALL(H_curr = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(H_curr);
  CV_CALL(WX_curr = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(WX_curr);
  CV_CALL(WH_curr = cvCreateMat( n_outputs, batch_size, CV_32F )); cvZero(WH_curr);
  CV_CALL(dE_dY_curr = cvCreateMat( n_outputs, batch_size, CV_32F )); cvZero(dE_dY_curr);
  CV_CALL(dE_dY_afder = cvCreateMat( n_outputs, batch_size, CV_32F )); cvZero(dE_dY_afder);
  CV_CALL(dH_curr = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(dH_curr);
  CV_CALL(dH_next = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(dH_next);
  CV_CALL(dH_raw  = cvCreateMat( n_hiddens, batch_size, CV_32F )); cvZero(dH_raw);
  CV_CALL(dWxh = cvCreateMat( layer_Wxh->rows, layer_Wxh->cols, CV_32F )); cvZero(dWxh);
  CV_CALL(dWhh = cvCreateMat( layer_Whh->rows, layer_Whh->cols, CV_32F )); cvZero(dWhh);
  CV_CALL(dWhy = cvCreateMat( layer_Why->rows, layer_Why->cols, CV_32F )); cvZero(dWhy);

  // bias on last column vector
  CV_CALL(cvGetCols( layer_Whh, &Whh_submat, 0, layer_Whh->cols-1));
  CV_CALL(cvGetCols( layer_Why, &Why_submat, 0, layer_Why->cols-1));
  CV_CALL(cvGetCol(  layer_Whh, &hbiascol,      layer_Whh->cols-1));
  CV_CALL(cvGetCol(  layer_Why, &ybiascol,      layer_Why->cols-1));
  CvMat * hbias = cvCreateMat(hbiascol.rows,batch_size,CV_32F); cvRepeat(&hbiascol,hbias);
  CvMat * ybias = cvCreateMat(ybiascol.rows,batch_size,CV_32F); cvRepeat(&ybiascol,ybias);
  CV_CALL(cvGetCols( dWhh, &dWhh_submat, 0, dWhh->cols-1));
  CV_CALL(cvGetCols( dWhy, &dWhy_submat, 0, dWhy->cols-1));
  CV_CALL(cvGetCol(  dWhh, &dhbiascol,      dWhh->cols-1));
  CV_CALL(cvGetCol(  dWhy, &dybiascol,      dWhy->cols-1));
  CvMat * dhbias = cvCreateMat(dhbiascol.rows,batch_size,CV_32F); cvRepeat(&dhbiascol,dhbias);
  CvMat * dybias = cvCreateMat(dybiascol.rows,batch_size,CV_32F); cvRepeat(&dybiascol,dybias);
  CV_CALL(cvGetCols( layer_dWhh, &layer_dWhh_submat, 0, layer_dWhh->cols-1));
  CV_CALL(cvGetCols( layer_dWhy, &layer_dWhy_submat, 0, layer_dWhy->cols-1));
  CV_CALL(cvGetCol(  layer_dWhh, &layer_dhbiascol,      layer_dWhh->cols-1));
  CV_CALL(cvGetCol(  layer_dWhy, &layer_dybiascol,      layer_dWhy->cols-1));
  CvMat * layer_dhbias = cvCreateMat(layer_dhbiascol.rows,batch_size,CV_32F); 
  CvMat * layer_dybias = cvCreateMat(layer_dybiascol.rows,batch_size,CV_32F);  
  cvRepeat(&layer_dhbiascol,layer_dhbias);
  cvRepeat(&layer_dybiascol,layer_dybias);

  // hidden states
  CvMat H_prev_hdr, H_curr_hdr, dH_curr_hdr, dH_next_hdr;
  CvMat WX_curr_hdr, WH_curr_hdr, dE_dY_curr_hdr;
  if (layer->time_index==seq_length-1){ cvZero(dH_next); }else{
    cvGetCol(layer_dH,&dH_next_hdr,layer->time_index+1); cvCopy(&dH_next_hdr,dH_next);
  }
  cvGetCol(layerH,&H_curr_hdr,layer->time_index); cvCopy(&H_curr_hdr,H_curr);
  cvGetCol(layer_WX,&WX_curr_hdr,layer->time_index); cvCopy(&WX_curr_hdr,WX_curr);
  cvGetCol(layer_WH,&WH_curr_hdr,layer->time_index); cvCopy(&WH_curr_hdr,WH_curr); 
  cvGetCol(layer_dH,&dH_curr_hdr,layer->time_index); // output variable
  CV_ASSERT(cvCountNonZero(WH_curr)>1);
  CV_ASSERT(layer_dE_dY->rows*layer_dE_dY->cols==seq_length*n_outputs*batch_size);
  CV_ASSERT(CV_MAT_TYPE(layer_dE_dY->type)==CV_32F);
  CvMat layer_dE_dY_reshaped = cvMat(seq_length,n_outputs,CV_32F,layer_dE_dY->data.ptr);
  CvMat * layer_dE_dY_transpose = cvCreateMat(n_outputs,seq_length,CV_32F);
  cvTranspose(&layer_dE_dY_reshaped,layer_dE_dY_transpose);
  cvGetCol(layer_dE_dY_transpose,&dE_dY_curr_hdr,layer->time_index); 
  cvCopy(&dE_dY_curr_hdr,dE_dY_curr);
  
  {CvScalar avg, sdv; cvAvgSdv(dE_dY_curr,&avg,&sdv); CV_ASSERT(sdv.val[0]>1e-5);}
    
  // compute (sig'(WX))*dE_dY
#if 0
  cvSigmoidDer(WH_curr,dE_dY_afder); // logistic regression
  cvMul(dE_dY_afder,dE_dY_curr,dE_dY_afder);
#else
  cvCopy(dE_dY_curr,dE_dY_afder);    // softmax for classification
  // cvSoftmaxDer(WH_curr,dE_dY_curr,dE_dY_afder);
#endif

  // dWhy += dE_dY_afder * H_curr'
  cvGEMM(dE_dY_afder,H_curr,1.f,0,1.f,&dWhy_submat,CV_GEMM_B_T);
  cvAdd(&layer_dWhy_submat,&dWhy_submat,&layer_dWhy_submat);
  // dby += dy
  CV_ASSERT(cvGetSize(&layer_dybiascol)==cvGetSize(dE_dY_afder));
  cvAdd(&layer_dybiascol,dE_dY_afder,&layer_dybiascol);

  // dH_curr = Why * dy + dH_next
  cvGEMM(&Why_submat,dE_dY_afder,1.f,dH_next,1.f,dH_curr,CV_GEMM_A_T);
  // dH_raw = tanh'(H_curr) * dH
  cvPow(H_curr,dH_raw,2.f); cvSubRS(dH_raw,cvScalar(1.f),dH_raw);
  cvMul(dH_raw, dH_curr, dH_raw);
  // dhbias += dH_raw
  cvAdd(&dhbiascol,dH_raw,&dhbiascol);

  // dWxh += dH_raw * X_curr'
  cvGEMM(dH_raw,X,1.f,0,1.f,dWxh,CV_GEMM_B_T);
  cvAdd(layer_dWxh,dWxh,layer_dWxh);
  // dWhh += dH_raw * H_prev'
  cvGEMM(dH_raw,H_prev,1.f,0,1.f,&dWhh_submat,CV_GEMM_B_T);
  cvAdd(&layer_dWhh_submat,&dWhh_submat,&layer_dWhh_submat);

  // gradient of hidden states, reserve it to continue backward pass to previous layers
  // dH_curr = Whh' * dH_raw
  CV_ASSERT(cvCountNAN(&dWhh_submat)<1 && cvCountNAN(dH_raw)<1);
  cvGEMM(&Whh_submat,dH_raw,1.f,0,1.f,dH_curr,CV_GEMM_A_T);
  cvCopy(dH_curr,&dH_curr_hdr); CV_ASSERT(cvCountNAN(&dH_curr_hdr)<1);

  // 2) update weights
  if (layer->time_index==0){
    float eta = -layer->init_learn_rate*cvInvSqrt(t);
    CvMat * W[5] = {layer_Wxh,&Whh_submat,&Why_submat,&hbiascol,&ybiascol};
    CvMat * dW[5] = {layer_dWxh,&layer_dWhh_submat,&layer_dWhy_submat,&dhbiascol,&dybiascol};
    for (int ii=0;ii<5;ii++){ 
      CV_ASSERT(cvCountNAN(dW[ii])<1 && cvCountNAN(W[ii])<1);
      cvMaxS(dW[ii],-5,dW[ii]); cvMinS(dW[ii],5,dW[ii]); 
      cvScaleAdd( dW[ii], cvScalar(eta), W[ii], W[ii] );
    }
  }

  if (n_output_layers){cvReleaseMat(&dE_dY);dE_dY=0;}
  if (WX){cvReleaseMat(&WX);WX=0;}
  if (WH){cvReleaseMat(&WH);WH=0;}
  if (H_prev){cvReleaseMat(&H_prev);H_prev=0;}
  if (H_curr){cvReleaseMat(&H_curr);H_curr=0;}
  if (WX_curr){cvReleaseMat(&WX_curr);WX_curr=0;}
  if (WH_curr){cvReleaseMat(&WH_curr);WH_curr=0;}
  if (dE_dY_curr){cvReleaseMat(&dE_dY_curr);dE_dY_curr=0;}
  if (dE_dY_afder){cvReleaseMat(&dE_dY_afder);dE_dY_afder=0;}
  if (dH_curr){cvReleaseMat(&dH_curr);dH_curr=0;}
  if (dH_next){cvReleaseMat(&dH_next);dH_next=0;}
  if (dH_raw ){cvReleaseMat(&dH_raw );dH_raw =0;}
  if (dWxh){cvReleaseMat(&dWxh);dWxh=0;}
  if (dWhh){cvReleaseMat(&dWhh);dWhh=0;}
  if (dWhy){cvReleaseMat(&dWhy);dWhy=0;}

  __END__;
}
