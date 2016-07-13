/*M////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to
//  this license. If you do not agree to this license, do not download,
//  install, copy or use the software.
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
//  * The name of Intel Corporation may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// This software is provided by the copyright holders and contributors
// "as is" and any express or implied warranties, including, but not
// limited to, the implied warranties of merchantability and fitness for
// a particular purpose are disclaimed. In no event shall the
// Intel Corporation or contributors be liable for any direct, indirect,
// incidental, special, exemplary, or consequential damages (including,
// but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such
// damage.
//
//M*/

// #include "cvconfig.h"
// #include "opencv2/core/core_c.h"
// #include "opencv2/core/internal.hpp"
#include "_dnn.h"

// #include "precomp.hpp"

#include "cnn.h"
// #include "cvext.h"
// #include "_dnn.h"

/*************************************************************************\
 *               Auxilary functions declarations                         *
\*************************************************************************/

/*------------ functions for the CNN classifier --------------------*/
void icvCNNModelPredict(const CvDNNStatModel * cnn_model, const CvMat * image, CvMat * probs, const int batch_size );
void icvCNNModelUpdate(
        CvDNNStatModel* cnn_model, const CvMat* images, int tflag,
        const CvMat* responses, const CvStatModelParams* params,
        const CvMat* CV_DEFAULT(0), const CvMat* sample_idx CV_DEFAULT(0),
        const CvMat* CV_DEFAULT(0), const CvMat* CV_DEFAULT(0));
void icvCNNModelRelease( CvDNNStatModel** cnn_model );

void icvTrainNetwork( CvNetwork* network,
        const CvMat* images, const CvMat* responses, 
        int grad_estim_type, int max_iter, int start_iter, int batch_size);

/*-------------- functions for the CNN network -------------------------*/
void icvNetworkAddLayer( CvNetwork* network, CvDNNLayer* layer );
CvDNNLayer* icvNetworkGetLayer( CvNetwork* network, const char * name );
void icvNetworkRelease( CvNetwork** network );
void icvNetworkRead( CvNetwork * network, CvFileStorage * fs );
void icvNetworkWrite( CvNetwork * network, CvFileStorage * fs );
/* In all layer functions we denote input by X and output by Y, where
   X and Y are column-vectors, so that
   length(X)==<n_input_planes>*<input_height>*<input_width>,
   length(Y)==<n_output_planes>*<output_height>*<output_width>.
*/

/*--------------------------- utility functions -----------------------*/
float icvEvalAccuracy(CvDNNLayer * last_layer, CvMat * result, CvMat * expected);

/**************************************************************************\
 *                 Functions implementations                              *
\**************************************************************************/

void icvCheckNetwork(CvNetwork * network,
                              CvDNNStatModelParams * params,
                              int img_size,
                              char * cvFuncName)  
{                                              
  CvDNNLayer* first_layer, *layer, *last_layer;
  int n_layers, i;                             
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;                                    
  first_layer = last_layer = network->first_layer;                      
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !icvIsDNNLayer(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    last_layer = layer;                                            
    layer = layer->next_layer;                                     
  }                                                                
  if ( i == 0 || i != n_layers || first_layer->prev_layer || layer ){
    CV_ERROR( CV_StsNullPtr, "Invalid network" );
  }
  if (icvIsConvolutionLayer(first_layer)){
    if ( first_layer->n_input_planes != 1 ) {                  
      CV_ERROR( CV_StsBadArg, "First layer must contain only one input plane" );
    }
    if ( img_size != first_layer->input_height*first_layer->input_width ){
      CV_ERROR( CV_StsBadArg, "Invalid input sizes of the first layer" );
    }
  }
  if (icvIsSimpleRNNLayer(last_layer)){
    if ( params->etalons->cols != last_layer->n_output_planes*
         ((CvDNNSimpleRNNLayer*)last_layer)->seq_length*
         last_layer->output_height*last_layer->output_width ) {
      CV_ERROR( CV_StsBadArg, "Invalid output sizes of the last layer" );
    }
  }else{
    if ( params->etalons->cols != last_layer->n_output_planes* 
         last_layer->output_height*last_layer->output_width ) {
      CV_ERROR( CV_StsBadArg, "Invalid output sizes of the last layer" );
    }
  }
}

void icvCheckCNNModelParams(CvDNNStatModelParams * params,
                                   CvDNNStatModel * cnn_model,
                                   char * cvFuncName)
{                                                                 
  if ( !params ) {
    CV_ERROR( CV_StsNullPtr, "Null <params> pointer" );           
  }
  if ( !ICV_IS_MAT_OF_TYPE(params->etalons, CV_32FC1) ) {
    CV_ERROR( CV_StsBadArg, "<etalons> must be CV_32FC1 type" );  
  }
  if ( params->etalons->rows != cnn_model->cls_labels->cols ) {
    CV_ERROR( CV_StsBadArg, "Invalid <etalons> size" );
  }
  if ( params->grad_estim_type != CV_DNN_GRAD_ESTIM_RANDOM &&      
      params->grad_estim_type != CV_DNN_GRAD_ESTIM_BY_WORST_IMG ) {
    CV_ERROR( CV_StsBadArg, "Invalid <grad_estim_type>" );        
  }
  if ( params->start_iter < 0 ) {
    CV_ERROR( CV_StsBadArg, "Parameter <start_iter> must be positive or zero" ); 
  }
  if ( params->max_iter < 1 ) {
    params->max_iter = 1;
  }
}

/********************************************************************\
 *                    Classifier functions                          *
\********************************************************************/

ML_IMPL CvDNNStatModel*	cvCreateStatModel(int flag, int size)// ,
    // CvDNNStatModelRelease release,
		// CvDNNStatModelPredict predict,
		// CvDNNStatModelUpdate update)
{
  CvDNNStatModel *p_model;
  CV_FUNCNAME("cvCreateStatModel");
  __CV_BEGIN__;
  //add the implementation here
  CV_CALL(p_model = (CvDNNStatModel*) cvAlloc(sizeof(*p_model)));
  memset(p_model, 0, sizeof(*p_model));
  p_model->release = icvCNNModelRelease; //release;
  p_model->update = icvCNNModelUpdate;// NULL;
  p_model->predict = icvCNNModelPredict;// NULL;
  __CV_END__;
  if (cvGetErrStatus() < 0) {
    CvDNNStatModel* base_ptr = (CvDNNStatModel*) p_model;
    if (p_model && p_model->release) {
      p_model->release(&base_ptr);
    }else{
      cvFree(&p_model);
    }
    p_model = 0;
  }
  return (CvDNNStatModel*) p_model;
  return NULL;
}

ML_IMPL CvDNNStatModel*
cvTrainCNNClassifier( const CvMat* _train_data, int tflag,
            const CvMat* _responses,
            const CvDNNStatModelParams* _params, 
            const CvMat*, const CvMat* _sample_idx, const CvMat*, const CvMat* )
{
  CvDNNStatModel* cnn_model    = 0;
  CvMat * train_data = cvCloneMat(_train_data);
  CvMat* responses             = 0;

  CV_FUNCNAME("cvTrainCNNClassifier");
  __BEGIN__;

  int n_images;
  int img_size;
  CvDNNStatModelParams* params = (CvDNNStatModelParams*)_params;

  CV_CALL(cnn_model = (CvDNNStatModel*)
    cvCreateStatModel(CV_STAT_MODEL_MAGIC_VAL|CV_DNN_MAGIC_VAL, sizeof(CvDNNStatModel)));
  
  img_size = _train_data->cols;
  n_images = _train_data->rows;
  cnn_model->cls_labels = params->cls_labels;
  responses = cvCreateMat(n_images,_responses->cols,CV_32F);
  cvConvert(_responses,responses);
  CV_ASSERT(CV_MAT_TYPE(train_data->type)==CV_32F);

  // normalize image value range
  if (icvIsConvolutionLayer(params->network->first_layer->next_layer)){
    double minval, maxval;
    cvMinMaxLoc(train_data,&minval,&maxval,0,0);
    cvSubS(train_data,cvScalar(minval),train_data);
    cvScale(train_data,train_data,10./((maxval-minval)*.5f));
    cvAddS(train_data,cvScalar(-1.f),train_data);
  }

  icvCheckCNNModelParams(params,cnn_model,cvFuncName);
  icvCheckNetwork(params->network,params,img_size,cvFuncName);

  cnn_model->network = params->network;
  CV_CALL(cnn_model->etalons = cvCloneMat( params->etalons ));

  CV_CALL( icvTrainNetwork( cnn_model->network, train_data, responses,
                              params->grad_estim_type, 
                              params->max_iter, params->start_iter, params->batch_size ));
  __END__;

  if ( cvGetErrStatus() < 0 && cnn_model ){
    cnn_model->release( (CvDNNStatModel**)&cnn_model );
  }
  cvReleaseMat( &train_data );
  cvReleaseMat( &responses );

  return (CvDNNStatModel*)cnn_model;
}

/*************************************************************************/
void icvTrainNetwork( CvNetwork* network,const CvMat* images, const CvMat* responses,
                               int grad_estim_type, int max_iter, int start_iter, int batch_size )
{
  CvMat** X     = 0;
  CvMat** dE_dX = 0;
  const int n_layers = network->n_layers;
  int k;

  CV_FUNCNAME("icvTrainNetwork");
  __BEGIN__;

  CvDNNLayer * first_layer = network->first_layer;
  CvDNNLayer * last_layer = cvGetCNNLastLayer(network);
  const int n_inputs   =
    first_layer->n_input_planes*first_layer->input_width*first_layer->input_height;
  const int n_images   = responses->rows; CV_ASSERT(n_images==images->rows);
  CvMat * X0 = cvCreateMat( batch_size*first_layer->seq_length, n_inputs, CV_32FC1 );
  CvDNNLayer* layer;
  int n;
  CvRNG rng = cvRNG(-1);

  CV_CALL(X = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  CV_CALL(dE_dX = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  memset( X, 0, (n_layers+1)*sizeof(CvMat*) );
  memset( dE_dX, 0, (n_layers+1)*sizeof(CvMat*) );

  // initialize input data
  CV_CALL(X[0] = cvCreateMat( batch_size*first_layer->seq_length, n_inputs, CV_32F ));
  CV_CALL(dE_dX[0] = cvCreateMat( batch_size*first_layer->seq_length, X[0]->cols, CV_32F ));
  cvZero(X[0]); cvZero(dE_dX[0]); cvZero(X0);
  for ( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer ){
    int n_outputs = layer->n_output_planes*layer->output_height*layer->output_width;
    if (icvIsInputLayer(layer)){
      CV_CALL(X[k+1] = cvCreateMat( batch_size*layer->seq_length, n_outputs, CV_32F )); 
      CV_CALL(dE_dX[k+1] = cvCreateMat( batch_size*layer->seq_length, X[k+1]->cols, CV_32F ));
    }else{
      CV_CALL(X[k+1] = cvCreateMat( batch_size, n_outputs, CV_32F )); 
      CV_CALL(dE_dX[k+1] = cvCreateMat( batch_size, X[k+1]->cols, CV_32F ));
    }
    cvZero(X[k+1]); cvZero(dE_dX[k+1]);
  }

  for ( n = 1; n <= max_iter; n++ )
  {
    float loss, max_loss = 0;
    int i;
    int nclasses = X[n_layers]->cols;
    CvMat * worst_img_idx = cvCreateMat(batch_size,1,CV_32S);
    int * right_etal_idx = responses->data.i;
    CvMat * etalon = cvCreateMat(batch_size*last_layer->seq_length,nclasses,CV_32F);

    // Use the random image
    if (first_layer->seq_length>1 && n_inputs!=images->cols){
      cvRandArr(&rng,worst_img_idx,CV_RAND_UNI,cvScalar(0),cvScalar(n_images-first_layer->seq_length));
    }else{
      cvRandArr(&rng,worst_img_idx,CV_RAND_UNI,cvScalar(0),cvScalar(n_images-1));
    }
    // cvPrintf(stderr,"%d, ",worst_img_idx);

    // 1) Compute the network output on the <X0>
    CV_ASSERT(CV_MAT_TYPE(X0->type)==CV_32F && CV_MAT_TYPE(images->type)==CV_32F);
    CV_ASSERT(n_inputs==X0->cols && n_inputs*first_layer->seq_length==images->cols);
    for ( k = 0; k < batch_size; k++ ){
      memcpy(X0->data.fl+images->cols*k,
             images->data.fl+images->cols*worst_img_idx->data.i[k],
             sizeof(float)*images->cols);
    }
    CV_CALL(cvCopy( X0, X[0] ));

    // Perform prediction with current weight parameters
    for ( k = 0, layer = first_layer; k < n_layers - 1; k++, layer = layer->next_layer ){
      CV_CALL(layer->forward( layer, X[k], X[k+1] )); 
    }
    CV_CALL(layer->forward( layer, X[k], X[k+1] ));

    // 2) Compute the gradient
    CvMat etalon_src, etalon_dst;
    CV_ASSERT(cvCountNAN(X[n_layers])<1);
    cvCopy( X[n_layers], dE_dX[n_layers] );
    for ( k = 0; k < batch_size; k++ ){
      cvGetRow(responses,&etalon_src,worst_img_idx->data.i[k]);
      CvMat etalon_reshaped_hdr;
      cvReshape(etalon,&etalon_reshaped_hdr,0,batch_size);
      CvMat * etalon_reshaped = cvCloneMat(&etalon_reshaped_hdr);
      cvGetRow(etalon_reshaped,&etalon_dst,k);
      cvCopy(&etalon_src, &etalon_dst);
      cvCopy(etalon_reshaped,&etalon_reshaped_hdr);
      cvReleaseMat(&etalon_reshaped);
    }
    cvSub( dE_dX[n_layers], etalon, dE_dX[n_layers] );

    // 3) Update weights by the gradient descent
    for ( k = n_layers; k > 0; k--, layer = layer->prev_layer ){
      CV_CALL(layer->backward( layer, n + start_iter, X[k-1], dE_dX[k], dE_dX[k-1] ));
    }

    // 4) compute loss & accuracy, print progress
    CvMat * etalon_transpose = cvCreateMat(etalon->cols,etalon->rows,CV_32F);
    cvTranspose(etalon,etalon_transpose);
    CvMat * Xn_transpose = 
      cvCreateMat(last_layer->n_output_planes,batch_size*last_layer->seq_length,CV_32F);
    cvTranspose(X[n_layers],Xn_transpose);
    float trloss = cvNorm(X[n_layers], etalon)/float(batch_size);
    float top1 = icvEvalAccuracy(last_layer, Xn_transpose, etalon_transpose);
    static double sumloss = 0; sumloss += trloss;
    static double sumacc  = 0; sumacc  += top1;
    if (int(float(n*100)/float(max_iter))<int(float((n+1)*100)/float(max_iter))){
      fprintf(stderr, "%d/%d = %.0f%%,",n+1,max_iter,float(n*100.f)/float(max_iter));
      fprintf(stderr, "sumacc: %.1f%%[%.1f%%], sumloss: %f\n", sumacc/float(n),top1,sumloss/float(n));
      if (n_inputs<100){fprintf(stderr,"input:\n");cvPrintf(stderr,"%.0f ", X[0]);}
      {fprintf(stderr,"output:\n");cvPrintf(stderr,"%.1f ", X[n_layers]);}
      {fprintf(stderr,"expect:\n");cvPrintf(stderr,"%.1f ", etalon);}
    }
    cvReleaseMat(&Xn_transpose);
    cvReleaseMat(&etalon_transpose);

    if (etalon){cvReleaseMat(&etalon);etalon=0;}
  }
  if (X0){cvReleaseMat(&X0);X0=0;}
  __END__;

  for ( k = 0; k <= n_layers; k++ ){
    cvReleaseMat( &X[k] );
    cvReleaseMat( &dE_dX[k] );
  }
  if (X){cvFree( &X );X=0;}
  if (dE_dX){cvFree( &dE_dX );dE_dX=0;}
}

float icvEvalAccuracy(CvDNNLayer * last_layer, CvMat * result, CvMat * expected)
{
  CV_FUNCNAME("icvEvalAccuracy");
  float top1 = 0;
  int n_outputs = last_layer->n_output_planes;
  int seq_length = icvIsSimpleRNNLayer(last_layer)?
    ((CvDNNSimpleRNNLayer*)last_layer)->seq_length:1;
  int batch_size = result->rows;
  CvMat * sorted = cvCreateMat(result->rows,result->cols,CV_32F);
  CvMat * indices = cvCreateMat(result->rows,result->cols,CV_32S);
  CvMat * indtop1 = cvCreateMat(result->rows,1,CV_32S);
  CvMat * expectedmat = cvCreateMat(result->rows,1,CV_32S);
  CvMat * indtop1true = cvCreateMat(result->rows,result->cols,CV_32S);
  CvMat * indtop1res = cvCreateMat(result->rows,1,CV_8U);
  __BEGIN__;
  cvSort(result,sorted,indices,CV_SORT_DESCENDING|CV_SORT_EVERY_ROW);
  cvGetCol(indices,indtop1,0);
  cvSort(expected,0,indtop1true,CV_SORT_DESCENDING|CV_SORT_EVERY_ROW);
  CV_ASSERT( CV_MAT_TYPE(indtop1true->type) == CV_32S && CV_MAT_TYPE(expectedmat->type) == CV_32S );
  for (int ii=0;ii<indtop1true->rows;ii++){
    CV_MAT_ELEM(*expectedmat,int,ii,0)=CV_MAT_ELEM(*indtop1true,int,ii,0);
  }
  cvCmp(indtop1,expectedmat,indtop1res,CV_CMP_EQ);
  top1=cvSum(indtop1res).val[0]*100.f/float(batch_size)/255.f;
  __END__;
  cvReleaseMat(&sorted);
  cvReleaseMat(&indices);
  cvReleaseMat(&indtop1);
  cvReleaseMat(&expectedmat);
  cvReleaseMat(&indtop1true);
  cvReleaseMat(&indtop1res);
  return top1;
}

/*************************************************************************/
void icvCNNModelPredict( const CvDNNStatModel* model, const CvMat* testdata, CvMat* result,
                                const int batch_size )
{
  CV_FUNCNAME("icvCNNModelPredict");
  __BEGIN__;

  CvDNNStatModel * cnn_model = (CvDNNStatModel*)model;
  CvDNNLayer * layer = 0;
  CvMat ** X = 0;
  int nclasses, i, k;
  float loss, min_loss = FLT_MAX;
  float* result_data;
  CvMat etalon;
  int nsamples = testdata->rows;;
  const CvDNNLayer * first_layer = cnn_model->network->first_layer;
  const CvDNNLayer * last_layer = cnn_model->network->get_last_layer(cnn_model->network);
  const int n_inputs   =
    first_layer->n_input_planes*first_layer->input_width*first_layer->input_height;

  if ( model==0 ) { CV_ERROR( CV_StsBadArg, "Invalid model" ); }

  nclasses = result->rows;
  const int n_layers = cnn_model->network->n_layers;
  const int seq_length = first_layer->seq_length;
  
  CvMat * samples = cvCloneMat(testdata);

  // normalize image value range
  if (icvIsConvolutionLayer(cnn_model->network->first_layer->next_layer)){
    double minval, maxval;
    cvMinMaxLoc(samples,&minval,&maxval,0,0);
    cvAddS(samples,cvScalar(-minval),samples);
    cvScale(samples,samples,10./((maxval-minval)*.5f));
    cvAddS(samples,cvScalar(-1.f),samples);
  }

  CvMat samples_reshape_hdr;
  cvReshape(samples,&samples_reshape_hdr,0,nsamples*first_layer->seq_length);

  CV_CALL(X = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  memset( X, 0, (n_layers+1)*sizeof(CvMat*) );

  int sidx=0; CvMat X0_hdr,Xn_hdr;
  
  // split full test data set into mini batches
  X[0] = cvCreateMat( batch_size*first_layer->seq_length, n_inputs, CV_32F ); cvZero(X[0]);
  for ( k = 0, layer = (CvDNNLayer*)first_layer; k < n_layers; k++, layer = layer->next_layer ){
    int n_outputs = layer->n_output_planes*layer->output_height*layer->output_width;
    if (icvIsInputLayer(layer)){
      CV_CALL(X[k+1] = cvCreateMat( batch_size*layer->seq_length, n_outputs, CV_32F ));
    }else{
      CV_CALL(X[k+1] = cvCreateMat( batch_size, n_outputs, CV_32F ));
    }cvZero(X[k+1]);
  }
  for (sidx=0;sidx<nsamples-batch_size;sidx+=batch_size){
    cvGetRows( &samples_reshape_hdr, &X0_hdr, sidx, sidx+batch_size ); cvCopy(&X0_hdr,X[0]);
    cvGetRows( result,               &Xn_hdr, sidx, sidx+batch_size );
    for ( k = 0, layer = (CvDNNLayer*)first_layer; k < n_layers; k++, layer = layer->next_layer ) {
      CV_CALL(layer->forward( layer, X[k], X[k+1] ));
    }cvCopy(X[n_layers],&Xn_hdr);
  }
  for ( k = 0; k <= n_layers; k++ ) { cvReleaseMat( &X[k] ); }

  // rest of the data set that can't divide by given batch_size
  int bsize = nsamples-sidx;
  X[0] = cvCreateMat( bsize*first_layer->seq_length, n_inputs, CV_32F ); cvZero(X[0]);
  for ( k = 0, layer = (CvDNNLayer*)first_layer; k < n_layers; k++, layer = layer->next_layer ){
    int n_outputs = layer->n_output_planes*layer->output_height*layer->output_width;
    if (icvIsInputLayer(layer)){
      CV_CALL(X[k+1] = cvCreateMat( bsize*layer->seq_length, n_outputs, CV_32F ));
    }else{
      CV_CALL(X[k+1] = cvCreateMat( bsize, n_outputs, CV_32F ));
    }cvZero(X[k+1]);
  }
  cvGetRows( &samples_reshape_hdr, &X0_hdr, sidx, sidx+bsize ); cvCopy(&X0_hdr,X[0]);
  cvGetRows( result,               &Xn_hdr, sidx, sidx+bsize );
  for ( k = 0, layer = (CvDNNLayer*)first_layer; k < n_layers; k++, layer = layer->next_layer ) {
    CV_CALL(layer->forward( layer, X[k], X[k+1] ));
  }cvCopy(X[n_layers],&Xn_hdr);

  cvReleaseMat(&samples);
  for ( k = 0; k <= n_layers; k++ ) { cvReleaseMat( &X[k] ); }
  if (X){cvFree( &X );X=0;}

  __END__;
}

/****************************************************************************************/
void icvCNNModelUpdate(
        CvDNNStatModel* _cnn_model, const CvMat* _train_data, int tflag,
        const CvMat* _responses, const CvStatModelParams* _params,
        const CvMat*, const CvMat* _sample_idx,
        const CvMat*, const CvMat* )
{
    const float** out_train_data = 0;
    CvMat* responses             = 0;
    CvMat* cls_labels            = 0;

    CV_FUNCNAME("icvCNNModelUpdate");
    __BEGIN__;

    int n_images, n_inputs, i;
    CvDNNStatModelParams* params = (CvDNNStatModelParams*)_params;
    CvDNNStatModel* cnn_model = (CvDNNStatModel*)_cnn_model;

    if ( cnn_model==0 ) {
        CV_ERROR( CV_StsBadArg, "Invalid model" );
    }

    // CV_CALL(cvPrepareTrainData( "cvTrainCNNClassifier",
    //     _train_data, tflag, _responses, CV_VAR_CATEGORICAL,
    //     0, _sample_idx, false, &out_train_data,
    //     &n_images, &n_inputs, &n_inputs, &responses,
    //     &cls_labels, 0, 0 ));

    // ICV_CHECK_CNN_MODEL_PARAMS(params);
    icvCheckCNNModelParams(params,cnn_model,cvFuncName);

    // Number of classes must be the same as when classifiers was created
    if ( !CV_ARE_SIZES_EQ(cls_labels, cnn_model->cls_labels) ) {
        CV_ERROR( CV_StsBadArg, "Number of classes must be left unchanged" );
    }
    for ( i = 0; i < cls_labels->cols; i++ ) {
      if ( cls_labels->data.i[i] != cnn_model->cls_labels->data.i[i] ) {
            CV_ERROR( CV_StsBadArg, "Number of classes must be left unchanged" );
      }
    }

    CV_CALL( icvTrainNetwork( cnn_model->network, _train_data, responses,
        params->grad_estim_type, params->max_iter,
        params->start_iter, params->batch_size ));

    __END__;

    cvFree( &out_train_data );
    cvReleaseMat( &responses );
}

/*************************************************************************/
void icvCNNModelRelease( CvDNNStatModel** cnn_model )
{
    CV_FUNCNAME("icvCNNModelRelease");
    __BEGIN__;

    CvDNNStatModel* cnn;
    if ( !cnn_model )
        CV_ERROR( CV_StsNullPtr, "Null double pointer" );

    cnn = *(CvDNNStatModel**)cnn_model;

    cvReleaseMat( &cnn->cls_labels );
    cvReleaseMat( &cnn->etalons );
    cnn->network->release( &cnn->network );

    cvFree( &cnn );

    __END__;

}

/************************************************************************ \
 *                       Network functions                              *
\************************************************************************/
ML_IMPL CvNetwork* cvCreateNetwork( CvDNNLayer* first_layer )
{
    CvNetwork* network = 0;

    CV_FUNCNAME( "cvCreateNetwork" );
    __BEGIN__;

    if ( !icvIsDNNLayer(first_layer) )
        CV_ERROR( CV_StsBadArg, "Invalid layer" );

    CV_CALL(network = (CvNetwork*)cvAlloc( sizeof(CvNetwork) ));
    memset( network, 0, sizeof(CvNetwork) );

    network->first_layer    = first_layer;
    network->n_layers  = 1;
    network->release   = icvNetworkRelease;
    network->add_layer = icvNetworkAddLayer;
    network->get_layer = icvNetworkGetLayer;
    network->get_last_layer = cvGetCNNLastLayer;
    network->eval      = icvEvalAccuracy;
    network->read      = icvNetworkRead;
    network->write     = icvNetworkWrite;

    __END__;

    if ( cvGetErrStatus() < 0 && network )
        cvFree( &network );

    return network;

}

/***********************************************************************/
void icvNetworkAddLayer( CvNetwork* network, CvDNNLayer* layer )
{
  CV_FUNCNAME( "icvNetworkAddLayer" );
  __BEGIN__;

  CvDNNLayer* prev_layer;

  if ( network == NULL ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer" );
  }

  // prev_layer = network->first_layer;
  // while ( prev_layer->next_layer ) { prev_layer = prev_layer->next_layer; }
  prev_layer = cvGetCNNLastLayer(network);

  if ( icvIsDenseLayer(layer) ){
    if ( ((CvDNNDenseLayer*)layer)->input_layers.size()==0 && 
         layer->n_input_planes != prev_layer->output_width*prev_layer->output_height*
         prev_layer->n_output_planes ) {
      CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
    }
    if ( layer->input_height != 1 || layer->output_height != 1 ||
         layer->input_width != 1  || layer->output_width != 1 ) {
      CV_ERROR( CV_StsBadArg, "Invalid size of the new layer" );
    }
  }else if ( icvIsConvolutionLayer(layer) || icvIsMaxPoolingLayer(layer) ){
    if ( prev_layer->n_output_planes != layer->n_input_planes ||
         prev_layer->output_height   != layer->input_height ||
         prev_layer->output_width    != layer->input_width ) {
      CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
    }
  }else if ( icvIsSimpleRNNLayer(layer) ) {
    if ( layer->input_height != 1 || layer->output_height != 1 ||
         layer->input_width != 1  || layer->output_width != 1 ) {
      CV_ERROR( CV_StsBadArg, "Invalid size of the new layer" );
    }
  }else if ( icvIsSpatialTransformLayer(layer) ) {
  }else if ( icvIsTimeDistributedLayer(layer) ) {
  }else if ( icvIsInputLayer(layer) ) {
  }else if ( icvIsRepeatVectorLayer(layer) ) {
  }else if ( icvIsMergeLayer(layer) ) {
    CV_ASSERT(((CvDNNMergeLayer*)layer)->input_layers.size()>=1);
    CV_ASSERT(((CvDNNMergeLayer*)layer)->input_layers.size()<=100);
  }else{
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  layer->prev_layer = prev_layer;
  prev_layer->next_layer = layer;
  network->n_layers++;

  __END__;
}

CvDNNLayer* icvNetworkGetLayer( CvNetwork* network, const char * name )
{
  CV_FUNCNAME("icvGetCNNGetLayer");
  CvDNNLayer* first_layer, *layer, *last_layer, *target_layer=0;
  int n_layers, i;
  __BEGIN__;
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;
  first_layer = last_layer = network->first_layer;
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !icvIsDNNLayer(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    if (!strcmp(layer->name,name)){target_layer=layer;break;}
    last_layer = layer;
    layer = layer->next_layer;
  }
  __END__;
  return target_layer;
}

CvDNNLayer * cvGetCNNLastLayer(CvNetwork * network)
{
  CV_FUNCNAME("icvGetCNNLastLayer");
  CvDNNLayer* first_layer, *layer, *last_layer;
  int n_layers, i;
  __BEGIN__;
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;
  first_layer = last_layer = network->first_layer;
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !icvIsDNNLayer(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    last_layer = layer;
    layer = layer->next_layer;
  }
  __END__;
  return last_layer;
}

/*************************************************************************/
void icvNetworkRelease( CvNetwork** network_pptr )
{
    CV_FUNCNAME( "icvReleaseNetwork" );
    __BEGIN__;

    CvNetwork* network = 0;
    CvDNNLayer* layer = 0, *next_layer = 0;
    int k;

    if ( network_pptr == NULL )
        CV_ERROR( CV_StsBadArg, "Null double pointer" );
    if ( *network_pptr == NULL )
        return;

    network = *network_pptr;
    layer = network->first_layer;
    if ( layer == NULL )
        CV_ERROR( CV_StsBadArg, "CNN is empty (does not contain any layer)" );

    // k is the number of the layer to be deleted
    for ( k = 0; k < network->n_layers && layer; k++ )
    {
        next_layer = layer->next_layer;
        layer->release( &layer );
        layer = next_layer;
    }

    if ( k != network->n_layers || layer)
        CV_ERROR( CV_StsBadArg, "Invalid network" );

    cvFree( &network );

    __END__;
}

/*************************************************************************\
 *                        Layer functions                                *
\*************************************************************************/
CvDNNLayer* icvCreateLayer( int layer_type, 
    const int dtype, const char * name, int header_size,
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int output_height, int output_width,
    float init_learn_rate, int learn_rate_decrease_type,
    CvDNNLayerRelease release, CvDNNLayerForward forward, CvDNNLayerBackward backward )
{
  CvDNNLayer* layer = 0;

  CV_FUNCNAME("icvCreateLayer");
  __BEGIN__;

  CV_ASSERT( release && forward && backward )
  CV_ASSERT( header_size >= sizeof(CvDNNLayer) )

  if ( n_input_planes < 1 || n_output_planes < 1 ||
       input_height   < 1 || input_width < 1 ||
       output_height  < 1 || output_width < 1 ||
       input_height < output_height || input_width  < output_width ) 
  {
    CV_ERROR( CV_StsBadArg, "Incorrect input or output parameters" );
  }
  if ( init_learn_rate < FLT_EPSILON ) {
    CV_ERROR( CV_StsBadArg, "Initial learning rate must be positive" );
  }
  if ( learn_rate_decrease_type != CV_DNN_LEARN_RATE_DECREASE_HYPERBOLICALLY &&
       learn_rate_decrease_type != CV_DNN_LEARN_RATE_DECREASE_SQRT_INV &&
       learn_rate_decrease_type != CV_DNN_LEARN_RATE_DECREASE_LOG_INV ) 
  {
    CV_ERROR( CV_StsBadArg, "Invalid type of learning rate dynamics" );
  }

  CV_CALL(layer = (CvDNNLayer*)cvAlloc( header_size ));
  memset( layer, 0, header_size );

  layer->flags = ICV_DNN_LAYER|layer_type;
  CV_ASSERT( icvIsDNNLayer(layer) );

  layer->dtype = dtype;
  strcpy(layer->name,name);

  layer->n_input_planes = n_input_planes;
  layer->input_height   = input_height;
  layer->input_width    = input_width;

  layer->n_output_planes = n_output_planes;
  layer->output_height   = output_height;
  layer->output_width    = output_width;

  layer->init_learn_rate = init_learn_rate;
  layer->decay_type = learn_rate_decrease_type;

  layer->release  = release;
  layer->forward  = forward;
  layer->backward = backward;

  __END__;

  if ( cvGetErrStatus() < 0 && layer) { cvFree( &layer ); }

  return layer;
}

void icvVisualizeCNNLayer(CvDNNLayer * layer, const CvMat * Y)
{
  CV_FUNCNAME("icvVisualizeCNNLayer");
  int hh = layer->output_height;
  int ww = layer->output_width;
  int nplanes = layer->n_output_planes;
  int nsamples = Y->rows;
  CvMat * imgY = cvCreateMat(hh*nsamples,ww*nplanes,CV_32F);
  float * imgYptr = imgY->data.fl;
  float * Yptr = Y->data.fl;
  int imgYcols = imgY->cols;
  __BEGIN__;
  CV_ASSERT(imgY->cols*imgY->rows == Y->cols*Y->rows);
  for (int si=0;si<nsamples;si++){
  for (int pi=0;pi<nplanes;pi++){
    for (int yy=0;yy<hh;yy++){
    for (int xx=0;xx<ww;xx++){
      CV_MAT_ELEM(*imgY,float,hh*si+yy,ww*pi+xx)=CV_MAT_ELEM(*Y,float,si,hh*ww*pi+ww*yy+xx);
    }
    }
  }
  }
  fprintf(stderr,"%dx%d,",imgY->cols,imgY->rows);
  CV_SHOW(imgY);
  __END__;
  cvReleaseMat(&imgY);
}


/*************************************************************************\
 *                           Utility functions                           *
\*************************************************************************/
void cvTanh(CvMat * src, CvMat * dst)
{
  CV_FUNCNAME("cvTanh");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_MAT_TYPE(dst->type));
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = tanh(srcptr[ii]);
    }
  }else if (CV_MAT_TYPE(src->type)==CV_64F){
    double * srcptr = src->data.db;
    double * dstptr = dst->data.db;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = tanh(srcptr[ii]);
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}

void cvTanhDer(CvMat * src, CvMat * dst) {
  CV_FUNCNAME("cvTanhDer");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_MAT_TYPE(dst->type));
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = 1.f-pow(tanh(srcptr[ii]),2);
    }
  }else if (CV_MAT_TYPE(src->type)==CV_64F){
    double * srcptr = src->data.db;
    double * dstptr = dst->data.db;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = 1.f-pow(tanh(srcptr[ii]),2);
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}
  
/****************************************************************************************\
*                              Read/Write CNN classifier                                *
\****************************************************************************************/
int icvIsModel( const void* ptr )
{
  return (ptr!=0);
}

/****************************************************************************************/
void icvReleaseCNNModel( void** ptr )
{
  CV_FUNCNAME("icvReleaseCNNModel");
  __BEGIN__;

  if ( !ptr ) { CV_ERROR( CV_StsNullPtr, "NULL double pointer" ); }
  CV_ASSERT((*ptr)!=0);

  icvCNNModelRelease( (CvDNNStatModel**)ptr );

  __END__;
}

/****************************************************************************************/
CvDNNLayer* icvReadCNNLayer( CvFileStorage* fs, CvFileNode* node )
{
  CvDNNLayer* layer = 0;
  CvMat* weights    = 0;
  CvMat* connect_mask = 0;

  CV_FUNCNAME("icvReadCNNLayer");
  __BEGIN__;

  int n_input_planes, input_height, input_width;
  int n_output_planes, output_height, output_width;
  int learn_type, layer_type;
  float init_learn_rate;

  CV_CALL(n_input_planes  = cvReadIntByName( fs, node, "n_input_planes",  -1 ));
  CV_CALL(input_height    = cvReadIntByName( fs, node, "input_height",    -1 ));
  CV_CALL(input_width     = cvReadIntByName( fs, node, "input_width",     -1 ));
  CV_CALL(n_output_planes = cvReadIntByName( fs, node, "n_output_planes", -1 ));
  CV_CALL(output_height   = cvReadIntByName( fs, node, "output_height",   -1 ));
  CV_CALL(output_width    = cvReadIntByName( fs, node, "output_width",    -1 ));
  CV_CALL(layer_type      = cvReadIntByName( fs, node, "layer_type",      -1 ));

  CV_CALL(init_learn_rate = (float)cvReadRealByName( fs, node, "init_learn_rate", -1 ));
  CV_CALL(learn_type = cvReadIntByName( fs, node, "learn_rate_decrease_type", -1 ));
  CV_CALL(weights    = (CvMat*)cvReadByName( fs, node, "weights" ));

  if ( n_input_planes < 0  || input_height < 0  || input_width < 0 ||
       n_output_planes < 0 || output_height < 0 || output_width < 0 ||
       init_learn_rate < 0 || learn_type < 0 || layer_type < 0 || !weights ) {
    CV_ERROR( CV_StsParseError, "" );
  }

  // if ( layer_type == ICV_DNN_CONVOLUTION_LAYER ) {
  //   const int K = input_height - output_height + 1;
  //   if ( K <= 0 || K != input_width - output_width + 1 ) {
  //     CV_ERROR( CV_StsBadArg, "Invalid <K>" );
  //   }
  //   CV_CALL(connect_mask = (CvMat*)cvReadByName( fs, node, "connect_mask" ));
  //   if ( !connect_mask ) {
  //     CV_ERROR( CV_StsParseError, "Missing <connect mask>" );
  //   }
  //   CV_CALL(layer = cvCreateConvolutionLayer( "", 0,
  //     n_input_planes, input_height, input_width, n_output_planes, K,
  //     init_learn_rate, learn_type, connect_mask, weights ));
  // } else if ( layer_type == ICV_DNN_SUBSAMPLING_LAYER ){
  //   const int sub_samp_scale = input_height/output_height;
  //   if ( sub_samp_scale <= 0 || sub_samp_scale != input_width/output_width ) {
  //     CV_ERROR( CV_StsBadArg, "Invalid <sub_samp_scale>" );
  //   }
  //   CV_CALL(layer = cvCreateSubSamplingLayer( "", 0,
  //     n_input_planes, input_height, input_width, sub_samp_scale,
  //     init_learn_rate, learn_type, weights ));
  // } else if ( layer_type == ICV_DNN_FULLCONNECT_LAYER ){
  //   if ( input_height != 1  || input_width != 1 || output_height != 1 || output_width != 1 ) { 
  //     CV_ERROR( CV_StsBadArg, "" ); 
  //   }
  //   CV_CALL(layer = cvCreateDenseLayer( "", 0, 0, n_input_planes, n_output_planes,
  //     init_learn_rate, learn_type, "tanh", weights ));
  // } else {
  //   CV_ERROR( CV_StsBadArg, "Invalid <layer_type>" );
  // }

  __END__;

  if ( cvGetErrStatus() < 0 && layer )
      layer->release( &layer );

  cvReleaseMat( &weights );
  cvReleaseMat( &connect_mask );

  return layer;
}

/****************************************************************************************/
void icvWriteCNNLayer( CvFileStorage* fs, CvDNNLayer* layer )
{
  CV_FUNCNAME ("icvWriteCNNLayer");
  __BEGIN__;

  if ( !icvIsDNNLayer(layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }

  CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_MAP, "opencv-ml-cnn-layer" ));

  CV_CALL(cvWriteInt( fs, "n_input_planes",  layer->n_input_planes ));
  CV_CALL(cvWriteInt( fs, "input_height",    layer->input_height ));
  CV_CALL(cvWriteInt( fs, "input_width",     layer->input_width ));
  CV_CALL(cvWriteInt( fs, "n_output_planes", layer->n_output_planes ));
  CV_CALL(cvWriteInt( fs, "output_height",   layer->output_height ));
  CV_CALL(cvWriteInt( fs, "output_width",    layer->output_width ));
  CV_CALL(cvWriteInt( fs, "learn_rate_decrease_type", layer->decay_type));
  CV_CALL(cvWriteReal( fs, "init_learn_rate", layer->init_learn_rate ));
  CV_CALL(cvWrite( fs, "weights", layer->weights ));

  if ( icvIsConvolutionLayer( layer )){
    CvDNNConvolutionLayer* l = (CvDNNConvolutionLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_DNN_CONVOLUTION_LAYER ));
    CV_CALL(cvWrite( fs, "connect_mask", l->connect_mask ));
  }else if ( icvIsMaxPoolingLayer( layer ) ){
    CvDNNMaxPoolingLayer* l = (CvDNNMaxPoolingLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_DNN_MAXPOOLLING_LAYER ));
  }else if ( icvIsDenseLayer( layer ) ){
    CvDNNDenseLayer* l = (CvDNNDenseLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_DNN_FULLCONNECT_LAYER ));
  }else {
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  CV_CALL( cvEndWriteStruct( fs )); //"opencv-ml-cnn-layer"

  __END__;
}

/****************************************************************************************/
void icvNetworkRead( CvNetwork * network, CvFileStorage * fs )
{
  CV_FUNCNAME("icvReadCNNModel");
  __BEGIN__;
  const int n_layers = network->n_layers;
  CvDNNLayer * layer = network->first_layer;
  CvFileNode * root = cvGetRootFileNode( fs );
  for (int ii=0;ii<n_layers;ii++,layer=layer->next_layer){
    if (icvIsSimpleRNNLayer(layer)){
      CvDNNSimpleRNNLayer * rnnlayer = (CvDNNSimpleRNNLayer*)(layer->ref_layer?layer->ref_layer:layer);
      char xhstr[1024],hhstr[1024],hystr[1024];
      sprintf(xhstr,"%s_Wxh",rnnlayer->name);
      sprintf(hhstr,"%s_Whh",rnnlayer->name);
      sprintf(hystr,"%s_Why",rnnlayer->name);
      cvCopy((CvMat*)cvReadByName(fs,root,xhstr),rnnlayer->Wxh);
      cvCopy((CvMat*)cvReadByName(fs,root,hhstr),rnnlayer->Whh);
      cvCopy((CvMat*)cvReadByName(fs,root,hystr),rnnlayer->Why);
    }else{
      cvCopy((CvMat*)cvReadByName(fs,root,layer->name),layer->weights);
    }
  }
  __END__;
}

/****************************************************************************************/
// void icvWriteCNNModel( CvFileStorage* fs, const char* name, 
//                               const void* struct_ptr, CvAttrList attr)
void icvNetworkWrite( CvNetwork * network, CvFileStorage * fs )
{
  CV_FUNCNAME ("icvWriteNetwork");
  __BEGIN__;
  const int n_layers = network->n_layers;
  CvDNNLayer * layer = (CvDNNLayer*)network->first_layer;
  for (int ii=0;ii<n_layers;ii++,layer=layer->next_layer){
    if (icvIsSimpleRNNLayer(layer)){
      CvDNNSimpleRNNLayer * rnnlayer = (CvDNNSimpleRNNLayer*)layer;
      char xhstr[1024],hhstr[1024],hystr[1024];
      sprintf(xhstr,"%s_Wxh",rnnlayer->name);
      sprintf(hhstr,"%s_Whh",rnnlayer->name);
      sprintf(hystr,"%s_Why",rnnlayer->name);
      if (rnnlayer->Wxh){
        cvWrite(fs,xhstr,rnnlayer->Wxh);
        cvWrite(fs,hhstr,rnnlayer->Whh);
        cvWrite(fs,hystr,rnnlayer->Why);
      }else{CV_ASSERT(!rnnlayer->Wxh && !rnnlayer->Whh && !rnnlayer->Why);}
    }else{
      if (layer->weights){cvWrite(fs,layer->name,layer->weights);}
    }
  }
  __END__;
}

CVAPI(void) cvPrintMergeLayerOutput(CvDNNLayer * layer, CvMat * Y)
{
  
}

// static int icvRegisterCNNStatModelType()
// {
//   CvTypeInfo info;
//   info.header_size = sizeof( info );
//   info.is_instance = icvIsModel;
//   info.release = icvReleaseCNNModel;
//   info.read = icvNetworkRead;
//   info.write = icvNetworkWrite;
//   info.clone = NULL;
//   info.type_name = CV_TYPE_NAME_ML_CNN;
//   cvRegisterType( &info );
//   return 1;
// } // End of icvRegisterCNNStatModelType
// static int cnn = icvRegisterCNNStatModelType();

// End of file
