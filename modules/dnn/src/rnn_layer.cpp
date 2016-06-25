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

ML_IMPL CvDNNLayer* cvCreateSimpleRNNLayer( 
    const int dtype, const char * name, const CvDNNLayer * ref_layer, 
    int n_inputs, int n_outputs, int n_hiddens, int seq_length, int time_index, 
    float init_learn_rate, int update_rule, const char * activation, 
    CvMat * Wxh, CvMat * Whh, CvMat * Why )
{
  CvDNNSimpleRNNLayer* layer = 0;

  CV_FUNCNAME("cvCreateSimpleRNNLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }

  fprintf(stderr,"SimpleRNNLayer(%s): input(%d), hidden(%d), output(%d), "
          "seq_length(%d), time_index(%d)\n", name,
          n_inputs, n_hiddens, n_outputs, seq_length, time_index);
  
  CV_CALL(layer = (CvDNNSimpleRNNLayer*)icvCreateLayer( ICV_DNN_RECURRENTNN_LAYER, dtype, name, 
      sizeof(CvDNNSimpleRNNLayer), n_inputs, 1, 1, n_outputs, 1, 1,
      init_learn_rate, update_rule,
      icvCNNRecurrentRelease, icvCNNRecurrentForward, icvCNNRecurrentBackward ));

  layer->ref_layer = (CvDNNLayer*)ref_layer;
  layer->weights = 0; // we don't use this !
  layer->time_index = time_index;
  layer->seq_length = seq_length;
  layer->n_hiddens = n_hiddens;
  layer->Wxh = 0;
  layer->Whh = 0;
  layer->Why = 0;
  strcpy(layer->activation,activation);
  layer->H = 0;
  layer->Y = 0;
  layer->loss = 0;
  layer->dE_dY = 0;

  int n_hiddens = layer->n_hiddens;
  if (!ref_layer){
    CV_ASSERT(layer->time_index==0);
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
  }else{
    CV_ASSERT(layer->time_index>0);
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvReleaseMat( &layer->Wxh );
    cvReleaseMat( &layer->Whh );
    cvReleaseMat( &layer->Why );
    cvFree( &layer );
  }

  return (CvDNNLayer*)layer;
}

void icvCNNRecurrentRelease( CvDNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNRecurrentRelease");
  __BEGIN__;

  CvDNNSimpleRNNLayer* layer = 0;

  if ( !p_layer ) { CV_ERROR( CV_StsNullPtr, "Null double pointer" ); }

  layer = *(CvDNNSimpleRNNLayer**)p_layer;

  if ( !layer ) { return; }
  if ( !icvIsSimpleRNNLayer((CvDNNLayer*)layer) ) { 
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
void icvCNNRecurrentForward( CvDNNLayer* _layer, const CvMat* X, CvMat * Y) 
{
  CV_FUNCNAME("icvCNNRecurrentForward");
  if ( !icvIsSimpleRNNLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;

  CvDNNSimpleRNNLayer * layer = (CvDNNSimpleRNNLayer*)_layer;
  CvDNNSimpleRNNLayer * ref_layer = (CvDNNSimpleRNNLayer*)layer->ref_layer;
  CvMat Wxh_submat, Whh_submat, hbiascol, Why_submat, ybiascol;
  int time_index = layer->time_index;
  int seq_length = layer->seq_length;
  int n_inputs = layer->n_input_planes;//Y->rows;
  int n_outputs = layer->n_output_planes;//Y->rows;
  int n_hiddens = layer->n_hiddens;
  int batch_size = X->rows;
  CvMat * WX = 0, * WH = 0, * H_prev = 0, * H_curr = 0, * WX_curr, * WH_curr;

  CV_ASSERT(X->rows == batch_size && X->cols == layer->n_input_planes);

  // memory allocation
  if (!ref_layer){
    if (!layer->H && !layer->Y && !layer->WX && !layer->WH){
      layer->H = cvCreateMat( seq_length, n_hiddens * batch_size, CV_32F ); 
      layer->Y = cvCreateMat( seq_length, n_outputs * batch_size, CV_32F ); 
      layer->WX = cvCreateMat( seq_length, n_hiddens * batch_size, CV_32F );
      layer->WH = cvCreateMat( seq_length, n_outputs * batch_size, CV_32F );
    }
    cvZero(layer->H);cvZero(layer->Y);cvZero(layer->WX);cvZero(layer->WH);
  }
  CvMat * Wxh = ref_layer?ref_layer->Wxh:layer->Wxh;
  CvMat * Whh = ref_layer?ref_layer->Whh:layer->Whh;
  CvMat * Why = ref_layer?ref_layer->Why:layer->Why;
  CvMat * layerH = ref_layer?ref_layer->H:layer->H;
  CvMat * layerY = ref_layer?ref_layer->Y:layer->Y;
  CvMat * layerWX = ref_layer?ref_layer->WX:layer->WX;
  CvMat * layerWH = ref_layer?ref_layer->WH:layer->WH;
  CV_ASSERT(cvGetSize(layerH)==cvGetSize(layerWX));
  CV_CALL(WX = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero( WX );
  CV_CALL(WH = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero( WH );
  CV_CALL(H_prev = cvCreateMat( 1, n_hiddens * batch_size, CV_32F )); cvZero(H_prev);
  CV_CALL(H_curr = cvCreateMat( 1, n_hiddens * batch_size, CV_32F )); cvZero(H_curr);
  CV_CALL(WX_curr = cvCreateMat( 1, n_hiddens * batch_size, CV_32F )); cvZero( WX_curr );
  CV_CALL(WH_curr = cvCreateMat( 1, n_outputs * batch_size, CV_32F )); cvZero( WH_curr );
  
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
  if (layer->time_index==0){ 
#if 1
    cvZero(H_prev); // LOGW("previous state should be reused.");
#else
    cvGetRow(layerH,&H_prev_hdr,layer->seq_length-1); cvCopy(&H_prev_hdr,H_prev);
#endif
  }else{
    cvGetRow(layerH,&H_prev_hdr,layer->time_index-1); cvCopy(&H_prev_hdr,H_prev);
  }
  cvGetRow(layerWX,&WX_curr_hdr,layer->time_index); cvCopy(&WX_curr_hdr,WX_curr);
  cvGetRow(layerWH,&WH_curr_hdr,layer->time_index); cvCopy(&WH_curr_hdr,WH_curr);
  CvMat H_prev_reshaped = cvMat(batch_size, n_hiddens, CV_32F, H_prev->data.ptr);
  CvMat H_curr_reshaped = cvMat(batch_size, n_hiddens, CV_32F, H_curr->data.ptr);
  CvMat WX_curr_reshaped = cvMat(batch_size, n_hiddens, CV_32F, WX_curr->data.ptr);
  CvMat WH_curr_reshaped = cvMat(batch_size, n_outputs, CV_32F, WH_curr->data.ptr);
  
  // H_curr = Wxh * X_curr + ( Whh * H_prev + bh )
  CV_CALL(cvGEMM( X, Wxh, 1, 0, 1, WX, CV_GEMM_B_T ));
  CV_CALL(cvGEMM( &H_prev_reshaped, &Whh_submat, 1, hbias, 1, WH, CV_GEMM_B_T+CV_GEMM_C_T ));  
  cvAdd(WX,WH,&H_curr_reshaped);
  cvCopy(&H_curr_reshaped,&WX_curr_reshaped);
  
  // activation for hidden states, relu and tahn is preferred
  cvTanh( H_curr, H_curr ); 

  // get H, Y for current time_index, output Y_curr_hdr, H_curr_hdr
  cvGetRow(layerH,&H_curr_hdr,layer->time_index); cvCopy(H_curr,&H_curr_hdr);
  cvGetRow(layerY,&Y_curr_hdr,layer->time_index);
  CvMat * Y_curr = cvCreateMat(batch_size,n_outputs,CV_32F); cvZero(Y_curr);
  CvMat Y_curr_reshape_hdr; cvReshape(&Y_curr_hdr,&Y_curr_reshape_hdr,0,batch_size);
  CV_ASSERT(cvCountNAN(&Y_curr_hdr)<1);

  // Y = activate(Why * H + by)
  cvGEMM( &H_curr_reshaped, &Why_submat, 1, ybias, 1, Y_curr, CV_GEMM_B_T+CV_GEMM_C_T );
  cvCopy(Y_curr,&WH_curr_reshaped); 
  CV_ASSERT(cvCountNonZero(&WH_curr_reshaped)>1);
  CV_ASSERT(cvCountNAN(Y_curr)<1);
  CV_CALL(cvCopy(WH_curr,&WH_curr_hdr));          // copy to layer->WH

  // apply activation to output
  if (!strcmp(layer->activation,"sigmoid")){
    CV_CALL(cvSigmoid( &Y_curr_hdr, &Y_curr_hdr )); 
  }else if (!strcmp(layer->activation,"tanh")){
    CV_CALL(cvTanh( &Y_curr_hdr, &Y_curr_hdr )); 
  }else if (!strcmp(layer->activation,"relu")){
    CV_CALL(cvReLU( &Y_curr_hdr, &Y_curr_hdr )); 
  }else if (!strcmp(layer->activation,"softmax")){
    CV_ASSERT(Y_curr->cols==n_outputs && Y_curr->rows==batch_size);
    cvSoftmax(Y_curr,Y_curr);
  }else{
    CV_ERROR(CV_StsBadArg,"invalid output activation type for RNN layer, `softmax` is prefered.");
  }
  cvCopy(Y_curr,&Y_curr_reshape_hdr);
  CV_ASSERT(cvCountNAN(Y_curr)<1);
  if (Y_curr){cvReleaseMat(&Y_curr);Y_curr=0;}

  // copy layer->Y to output variable Y
#if 0
  cvTranspose(layerY,Y);
#else
  CvMat layer_Y_submat_hdr, Y_submat_hdr;
  for (int tidx=0;tidx<seq_length;tidx++){
  for (int bidx=0;bidx<batch_size;bidx++){
    cvGetSubRect(layerY,&layer_Y_submat_hdr,cvRect(bidx*n_outputs,tidx,n_outputs,1));
    cvGetSubRect(Y,&Y_submat_hdr,cvRect(0,seq_length*bidx+tidx,n_outputs,1));
    cvCopy(&layer_Y_submat_hdr,&Y_submat_hdr);
  }
  }
#endif
    
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
void icvCNNRecurrentBackward( CvDNNLayer* _layer, int t,
                                     const CvMat * X, const CvMat * _dE_dY, CvMat * dE_dX )
{
  CV_FUNCNAME( "icvCNNRecurrentBackward" );
  if ( !icvIsSimpleRNNLayer(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }

  __BEGIN__;

  CvDNNSimpleRNNLayer * layer = (CvDNNSimpleRNNLayer*)_layer;
  CvDNNSimpleRNNLayer * ref_layer = (CvDNNSimpleRNNLayer*)layer->ref_layer;
  CvMat * dE_dY = (CvMat*)_dE_dY;
  
  // TODO: compute average from all output_layers
  int n_output_layers = ref_layer?ref_layer->output_layers.size():layer->output_layers.size();
  const int batch_size = X->rows;
  if (n_output_layers){
    const int n_Y_planes = layer->n_output_planes;
    const int Y_plane_size   = layer->output_height*layer->output_width;
    dE_dY = cvCreateMat(batch_size,Y_plane_size*n_Y_planes,CV_32F); cvZero(dE_dY);
    for (int li=0;li<n_output_layers;li++){
      CvDNNLayer * output_layer = ref_layer?ref_layer->output_layers[li]:layer->output_layers[li];
      if (icvIsDenseLayer(output_layer)){
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
  CvMat  layer_Whh_submat,  layer_Why_submat, layer_hbiascol, layer_ybiascol;
  CvMat dWhh_submat, dWhy_submat; 
  CvMat layer_dWhh_submat, layer_dhbiascol, layer_dWhy_submat, layer_dybiascol;
  int time_index = layer->time_index;
  int seq_length = layer->seq_length;
  int n_inputs = layer->n_input_planes;
  int n_outputs = layer->n_output_planes;
  int n_hiddens = layer->n_hiddens;
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
                layer_dWhh==ref_layer->dWhh && layer_dWhy==ref_layer->dWhy);
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
  CV_CALL(WX = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero( WX );
  CV_CALL(WH = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero( WH );
  CV_CALL(H_prev = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(H_prev);
  CV_CALL(H_curr = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(H_curr);
  CV_CALL(WX_curr = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(WX_curr);
  CV_CALL(WH_curr = cvCreateMat( batch_size, n_outputs, CV_32F )); cvZero(WH_curr);
  CV_CALL(dE_dY_curr = cvCreateMat( batch_size, n_outputs, CV_32F )); cvZero(dE_dY_curr);
  CV_CALL(dE_dY_afder = cvCreateMat( batch_size, n_outputs, CV_32F )); cvZero(dE_dY_afder);
  CV_CALL(dH_curr = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(dH_curr);
  CV_CALL(dH_next = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(dH_next);
  CV_CALL(dH_raw  = cvCreateMat( batch_size, n_hiddens, CV_32F )); cvZero(dH_raw);
  // follow variables are added to layer_dWxh,layer_dWhh,layer_dWhy, 
  // and they are later added to layer_Wxh,layer_Whh,layer_Why
  CV_CALL(dWxh = cvCreateMat( layer_Wxh->rows, layer_Wxh->cols, CV_32F )); cvZero(dWxh);
  CV_CALL(dWhh = cvCreateMat( layer_Whh->rows, layer_Whh->cols, CV_32F )); cvZero(dWhh);
  CV_CALL(dWhy = cvCreateMat( layer_Why->rows, layer_Why->cols, CV_32F )); cvZero(dWhy);

  // bias on last column vector
  CV_CALL(cvGetCols( layer_Whh, &layer_Whh_submat, 0, layer_Whh->cols-1));
  CV_CALL(cvGetCols( layer_Why, &layer_Why_submat, 0, layer_Why->cols-1));
  CV_CALL(cvGetCol(  layer_Whh, &layer_hbiascol,      layer_Whh->cols-1));
  CV_CALL(cvGetCol(  layer_Why, &layer_ybiascol,      layer_Why->cols-1));
  CV_CALL(cvGetCols( dWhh, &dWhh_submat, 0, dWhh->cols-1));
  CV_CALL(cvGetCols( dWhy, &dWhy_submat, 0, dWhy->cols-1));
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
  CvMat dH_next_reshape_hdr, H_prev_reshape_hdr, H_curr_reshape_hdr, WX_curr_reshape_hdr, WH_curr_reshape_hdr;
  if (layer->time_index==seq_length-1){ cvZero(dH_next); }else{
    cvGetRow(layer_dH,&dH_next_hdr,layer->time_index+1); 
    cvReshape(dH_next,&dH_next_reshape_hdr,0,1); 
    cvCopy(&dH_next_hdr,&dH_next_reshape_hdr);
  }
  if (layer->time_index==0){
#if 1
    cvZero(H_prev); // LOGW("this should be properly handled.");
#else
    cvGetRow(layerH,&H_prev_hdr,layer->seq_length-1); 
    cvReshape(&H_prev_hdr,&H_prev_reshape_hdr,0,batch_size); cvCopy(&H_prev_reshape_hdr,H_prev);
#endif
  }else{
    cvGetRow(layerH,&H_prev_hdr,layer->time_index-1); 
    cvReshape(&H_prev_hdr,&H_prev_reshape_hdr,0,batch_size); cvCopy(&H_prev_reshape_hdr,H_prev);
  }
  cvGetRow(layerH,&H_curr_hdr,layer->time_index); 
  cvReshape(&H_curr_hdr,&H_curr_reshape_hdr,0,batch_size); cvCopy(&H_curr_reshape_hdr,H_curr);
  cvGetRow(layer_WX,&WX_curr_hdr,layer->time_index); 
  cvReshape(&WX_curr_hdr,&WX_curr_reshape_hdr,0,batch_size); cvCopy(&WX_curr_reshape_hdr,WX_curr);
  cvGetRow(layer_WH,&WH_curr_hdr,layer->time_index); 
  cvReshape(&WH_curr_hdr,&WH_curr_reshape_hdr,0,batch_size); cvCopy(&WH_curr_reshape_hdr,WH_curr); 
  cvGetRow(layer_dH,&dH_curr_hdr,layer->time_index); // output variable
  CV_ASSERT(cvCountNonZero(WH_curr)>1);

  // copy `layer_dE_dY` with current time_index, to variable `dE_dY_curr`
  CV_ASSERT(layer_dE_dY->rows*layer_dE_dY->cols==seq_length*n_outputs*batch_size);
  CV_ASSERT(CV_MAT_TYPE(layer_dE_dY->type)==CV_32F);
  CV_ASSERT(cvCountNAN(layer_dE_dY)<1);
  cvZero(dE_dY_curr);
  for (int bidx=0;bidx<batch_size;bidx++){
    CvMat layer_dE_dY_submat, dE_dY_curr_submat;
    cvGetRow(layer_dE_dY,&layer_dE_dY_submat,seq_length*bidx+time_index);
    cvGetRow(dE_dY_curr,&dE_dY_curr_submat,bidx);
    cvCopy(&layer_dE_dY_submat, &dE_dY_curr_submat);
  }

  // output activation derivative
  if (!strcmp(layer->activation,"sigmoid")){
    cvSigmoidDer(WH_curr,dE_dY_afder);
    cvMul(dE_dY_afder,dE_dY_curr,dE_dY_afder);
  }else if (!strcmp(layer->activation,"tanh")){
    cvTanhDer(WH_curr,dE_dY_afder);
    cvMul(dE_dY_afder,dE_dY_curr,dE_dY_afder);
  }else if (!strcmp(layer->activation,"relu")){
    cvReLUDer(WH_curr,dE_dY_afder);
    cvMul(dE_dY_afder,dE_dY_curr,dE_dY_afder);
  }else if (!strcmp(layer->activation,"softmax")){
    cvCopy(dE_dY_curr,dE_dY_afder);    // softmax for classification
  }else{
    CV_ERROR(CV_StsBadArg,"invalid output activation type for RNN layer, `softmax` is prefered.");
  }

  // dWhy += dE_dY_afder * H_curr'
  CV_GEMM(dE_dY_afder,H_curr,1.f,0,1.f,&dWhy_submat,CV_GEMM_A_T);
  if (time_index==seq_length-1){CV_ASSERT(cvSdv(&layer_dWhy_submat)<1e-5f);}
  else{CV_ASSERT(cvSdv(&layer_dWhy_submat)>1e-5f);}
  cvAdd(&layer_dWhy_submat,&dWhy_submat,&layer_dWhy_submat);
  
  // dby += dy
  CvMat * dE_dY_afder_transpose = cvCreateMat(dE_dY_afder->cols,dE_dY_afder->rows,CV_32F);
  cvTranspose(dE_dY_afder,dE_dY_afder_transpose);
  cvAdd(layer_dybias,dE_dY_afder_transpose,layer_dybias);
  cvReduce(layer_dybias,&layer_dybiascol,-1,CV_REDUCE_AVG); // ::TODO:: average dybias update ???
  cvZero(layer_dybias);
  cvReleaseMat(&dE_dY_afder_transpose);

  // dH_curr = Why * dy + dH_next
  CV_GEMM(dE_dY_afder,&layer_Why_submat,1.f,dH_next,1.f,dH_curr,0);
  // dH_raw = tanh'(H_curr) * dH = (1-H_curr.*H_curr)*dH
  cvPow(H_curr,dH_raw,2.f); cvSubRS(dH_raw,cvScalar(1.f),dH_raw);
  cvMul(dH_raw, dH_curr, dH_raw);
  // dhbias += dH_raw
  CvMat * dH_raw_transpose = cvCreateMat(dH_raw->cols,dH_raw->rows,CV_32F);
  cvTranspose(dH_raw,dH_raw_transpose);
  cvAdd(layer_dhbias,dH_raw_transpose,layer_dhbias);
  cvReduce(layer_dhbias,&layer_dhbiascol,-1,CV_REDUCE_AVG); // ::TODO:: average dhbias update ???
  cvZero(layer_dhbias);
  cvReleaseMat(&dH_raw_transpose);

  // dWxh += dH_raw * X_curr'
  CV_ASSERT(dH_raw->rows==batch_size && X->rows==batch_size);
  cvGEMM(dH_raw,X,1.f,0,1.f,dWxh,CV_GEMM_A_T);
  cvScale(dWxh,dWxh,1./batch_size); // batch normalize
  cvAdd(layer_dWxh,dWxh,layer_dWxh);
  // dWhh += dH_raw * H_prev'
  CV_ASSERT(dH_raw->rows==batch_size && H_prev->rows==batch_size);
  CV_GEMM(dH_raw,H_prev,1.f,0,1.f,&dWhh_submat,CV_GEMM_A_T);
  cvScale(&dWhh_submat,&dWhh_submat,1./batch_size); // batch normalize
  cvAdd(&layer_dWhh_submat,&dWhh_submat,&layer_dWhh_submat);

  // gradient of hidden states, reserve it to continue backward pass to previous layers
  // dH_curr = Whh' * dH_raw, while (A'*B)=(B'*A)'
  CV_ASSERT(cvCountNAN(&dWhh_submat)<1 && cvCountNAN(dH_raw)<1);
  CV_ASSERT(dH_raw->cols==n_hiddens && layer_Whh_submat.rows==n_hiddens);
  cvGEMM(dH_raw,&layer_Whh_submat,1.f,0,1.f,dH_curr,0);
  CvMat dH_curr_reshape_hdr; cvReshape(dH_curr,&dH_curr_reshape_hdr,0,1);
  cvCopy(&dH_curr_reshape_hdr,&dH_curr_hdr); CV_ASSERT(cvCountNAN(&dH_curr_hdr)<1);

  // 2) update weights
  if (layer->time_index==0){
    float eta = -layer->init_learn_rate*cvInvSqrt(t);
    CvMat * W[5] = {layer_Wxh,&layer_Whh_submat,&layer_Why_submat,&layer_hbiascol,&layer_ybiascol};
    CvMat * dW[5] = {layer_dWxh,&layer_dWhh_submat,&layer_dWhy_submat,
                     &layer_dhbiascol,&layer_dybiascol};
    for (int ii=0;ii<5;ii++){ 
      CV_ASSERT(cvCountNAN(dW[ii])<1 && cvCountNAN(W[ii])<1);
      cvMaxS(dW[ii],-5,dW[ii]); cvMinS(dW[ii],5,dW[ii]); 
      cvScaleAdd( dW[ii], cvScalar(eta), W[ii], W[ii] );
    }
  }

  if (n_output_layers){cvReleaseMat(&dE_dY);dE_dY=0;}

  if (layer_dhbias){cvReleaseMat(&layer_dhbias);layer_dhbias=0;}
  if (layer_dybias){cvReleaseMat(&layer_dybias);layer_dybias=0;}
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
