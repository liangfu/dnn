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

#include "precomp.hpp"

#if 1 

// #include "cnn.h"
#include "cvext.h"

// sigmoid function
// #define SIG(p) (1.7159*tanh(0.66666667*p))
// derivative of the sigmoid
// #define DSIG(p) (0.66666667/1.7159*(1.7159+(p))*(1.7159-(p)))  

/*************************************************************************\
 *               Auxilary functions declarations                         *
\*************************************************************************/
/*------------ functions for the CNN classifier --------------------*/
static float icvCNNModelPredict(
        const CvCNNStatModel* cnn_model,
        const CvMat* image,
        CvMat* probs CV_DEFAULT(0) );

static void icvCNNModelUpdate(
        CvCNNStatModel* cnn_model, const CvMat* images, int tflag,
        const CvMat* responses, const CvStatModelParams* params,
        const CvMat* CV_DEFAULT(0), const CvMat* sample_idx CV_DEFAULT(0),
        const CvMat* CV_DEFAULT(0), const CvMat* CV_DEFAULT(0));

static void icvCNNModelRelease( CvCNNStatModel** cnn_model );

static void icvTrainCNNetwork( CvCNNetwork* network,
                               const CvMat* images,
                               const CvMat* responses,
                               const CvMat* etalons,
                               int grad_estim_type,
                               int max_iter,
                               int start_iter,
                               int batch_size);

/*-------------- functions for the CNN network -------------------------*/
static void icvCNNetworkAddLayer( CvCNNetwork* network, CvCNNLayer* layer );
static CvCNNLayer* icvCNNetworkGetLayer( CvCNNetwork* network, const char * name );
static void icvCNNetworkRelease( CvCNNetwork** network );

/* In all layer functions we denote input by X and output by Y, where
   X and Y are column-vectors, so that
   length(X)==<n_input_planes>*<input_height>*<input_width>,
   length(Y)==<n_output_planes>*<output_height>*<output_width>.
*/
static void icvVisualizeCNNLayer(CvCNNLayer * layer, const CvMat * Y);
/*--------------- functions for convolutional layer --------------------*/
static void icvCNNConvolutionRelease( CvCNNLayer** p_layer );
static void icvCNNConvolutionForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNConvolutionBackward( CvCNNLayer*  layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX );

/*------------------ functions for sub-sampling layer -------------------*/
static void icvCNNSubSamplingRelease( CvCNNLayer** p_layer );
static void icvCNNSubSamplingForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNSubSamplingBackward( CvCNNLayer*  layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX );

/*---------- functions for full connected layer -----------------------*/
static void icvCNNFullConnectRelease( CvCNNLayer** p_layer );
static void icvCNNFullConnectForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNFullConnectBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*-------------- functions for image cropping layer ------------------*/
static void icvCNNImgCroppingRelease( CvCNNLayer** p_layer );
static void icvCNNImgCroppingForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNImgCroppingBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*-------------- functions for recurrent layer -----------------------*/
static void icvCNNRecurrentRelease( CvCNNLayer** p_layer );
static void icvCNNRecurrentForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNRecurrentBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*-------------- functions for recurrent layer -----------------------*/
static void icvCNNInputDataRelease( CvCNNLayer** p_layer );
static void icvCNNInputDataForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNInputDataBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX );

/*--------------------------- utility functions -----------------------*/
static float icvEvalAccuracy(CvMat * result, CvMat * expected);

/*------------------------ activation functions -----------------------*/
static void cvTanh(CvMat * src, CvMat * dst);
static void cvTanhDer(CvMat * src, CvMat * dst);
static void cvSigmoid(CvMat * src, CvMat * dst);
static void cvSigmoidDer(CvMat * src, CvMat * dst);
static void cvReLU(CvMat * src, CvMat * dst){cvMaxS( src, 0, dst );}
static void cvReLUDer(CvMat * src, CvMat * dst);

/**************************************************************************\
 *                 Functions implementations                              *
\**************************************************************************/

static void icvCheckCNNetwork(CvCNNetwork * network,
                              CvCNNStatModelParams * params,
                              int img_size,
                              char * cvFuncName)  
{                                              
  CvCNNLayer* first_layer, *layer, *last_layer;
  int n_layers, i;                             
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;                                    
  first_layer = last_layer = network->first_layer;                      
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !ICV_IS_CNN_LAYER(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    last_layer = layer;                                            
    layer = layer->next_layer;                                     
  }                                                                
  if ( i == 0 || i != n_layers || first_layer->prev_layer || layer ){
    CV_ERROR( CV_StsNullPtr, "Invalid network" );
  }
  if (ICV_IS_CNN_CONVOLUTION_LAYER(first_layer)){
    if ( first_layer->n_input_planes != 1 ) {                  
      CV_ERROR( CV_StsBadArg, "First layer must contain only one input plane" );
    }
    if ( img_size != first_layer->input_height*first_layer->input_width ){
      CV_ERROR( CV_StsBadArg, "Invalid input sizes of the first layer" );
    }
  }
  if (ICV_IS_CNN_RECURRENT_LAYER(last_layer)){
    if ( params->etalons->cols != last_layer->n_output_planes*
         ((CvCNNRecurrentLayer*)last_layer)->seq_length*
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

static void icvCheckCNNModelParams(CvCNNStatModelParams * params,
                                   CvCNNStatModel * cnn_model,
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
  if ( params->grad_estim_type != CV_CNN_GRAD_ESTIM_RANDOM &&      
      params->grad_estim_type != CV_CNN_GRAD_ESTIM_BY_WORST_IMG ) {
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

ML_IMPL CvCNNStatModel*	cvCreateCNNStatModel(int flag, int size)// ,
    // CvCNNStatModelRelease release,
		// CvCNNStatModelPredict predict,
		// CvCNNStatModelUpdate update)
{
  CvCNNStatModel *p_model;
  CV_FUNCNAME("cvCreateStatModel");
  __CV_BEGIN__;
  //add the implementation here
  CV_CALL(p_model = (CvCNNStatModel*) cvAlloc(sizeof(*p_model)));
  memset(p_model, 0, sizeof(*p_model));
  p_model->release = icvCNNModelRelease; //release;
  p_model->update = icvCNNModelUpdate;// NULL;
  p_model->predict = icvCNNModelPredict;// NULL;
  __CV_END__;
  if (cvGetErrStatus() < 0) {
    CvCNNStatModel* base_ptr = (CvCNNStatModel*) p_model;
    if (p_model && p_model->release) {
      p_model->release(&base_ptr);
    }else{
      cvFree(&p_model);
    }
    p_model = 0;
  }
  return (CvCNNStatModel*) p_model;
  return NULL;
}

ML_IMPL CvCNNStatModel*
cvTrainCNNClassifier( const CvMat* _train_data, int tflag,
            const CvMat* _responses,
            const CvCNNStatModelParams* _params, 
            const CvMat*, const CvMat* _sample_idx, const CvMat*, const CvMat* )
{
  CvCNNStatModel* cnn_model    = 0;
  CvMat * train_data = cvCloneMat(_train_data);
  CvMat* responses             = 0;

  CV_FUNCNAME("cvTrainCNNClassifier");
  __BEGIN__;

  int n_images;
  int img_size;
  CvCNNStatModelParams* params = (CvCNNStatModelParams*)_params;

  CV_CALL(cnn_model = (CvCNNStatModel*)
    cvCreateCNNStatModel(CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));
  
  img_size = _train_data->cols;
  n_images = _train_data->rows;
  cnn_model->cls_labels = params->cls_labels;
  responses = cvCreateMat(_responses->cols,n_images,CV_32S);
  cvTranspose(_responses,responses);
  CV_ASSERT(CV_MAT_TYPE(train_data->type)==CV_32F);

  // normalize image value range
  if (ICV_IS_CNN_CONVOLUTION_LAYER(params->network->first_layer)){
    double minval, maxval;
    cvMinMaxLoc(train_data,&minval,&maxval,0,0);
    cvSubS(train_data,cvScalar(minval),train_data);
    cvScale(train_data,train_data,10./((maxval-minval)*.5f));
    cvAddS(train_data,cvScalar(-1.f),train_data);
  }

  icvCheckCNNModelParams(params,cnn_model,cvFuncName);
  icvCheckCNNetwork(params->network,params,img_size,cvFuncName);

  cnn_model->network = params->network;
  CV_CALL(cnn_model->etalons = cvCloneMat( params->etalons ));

  CV_CALL( icvTrainCNNetwork( cnn_model->network, train_data, responses,
                              cnn_model->etalons, params->grad_estim_type, 
                              params->max_iter, params->start_iter, params->batch_size ));
  __END__;

  if ( cvGetErrStatus() < 0 && cnn_model ){
    cnn_model->release( (CvCNNStatModel**)&cnn_model );
  }
  cvReleaseMat( &train_data );
  cvReleaseMat( &responses );

  return (CvCNNStatModel*)cnn_model;
}

/*************************************************************************/
static void icvTrainCNNetwork( CvCNNetwork* network,const CvMat* images, const CvMat* responses,
                               const CvMat* etalons,
                               int grad_estim_type, int max_iter, int start_iter, int batch_size )
{
  CvMat** X     = 0;
  CvMat** dE_dX = 0;
  const int n_layers = network->n_layers;
  int k;

  CV_FUNCNAME("icvTrainCNNetwork");
  __BEGIN__;

  CvCNNLayer * first_layer = network->first_layer;
  CvCNNLayer * last_layer = cvGetCNNLastLayer(network);
  const int img_height = first_layer->input_height;
  const int img_width  = first_layer->input_width;
  const int img_size   = ICV_IS_CNN_INPUTDATA_LAYER(first_layer)?
      first_layer->n_input_planes*img_width*img_height*((CvCNNInputDataLayer*)first_layer)->seq_length:
      first_layer->n_input_planes*img_width*img_height;
  const int n_images   = responses->cols;
  CvMat * X0_transpose = cvCreateMat( batch_size, img_size, CV_32FC1 );
  CvCNNLayer* layer;
  int n;
  CvRNG rng = cvRNG(-1);

  CV_CALL(X = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  CV_CALL(dE_dX = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  memset( X, 0, (n_layers+1)*sizeof(CvMat*) );
  memset( dE_dX, 0, (n_layers+1)*sizeof(CvMat*) );

  // initialize input data
  CV_CALL(X[0] = cvCreateMat( img_size, batch_size, CV_32FC1 ));
  CV_CALL(dE_dX[0] = cvCreateMat( batch_size, X[0]->rows, CV_32FC1 ));
  for ( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer ){
    int n_outputs = // ICV_IS_CNN_RECURRENT_LAYER(layer)?layer->n_input_planes:
      layer->n_output_planes*layer->output_height*layer->output_width;
    if (layer==last_layer && ICV_IS_CNN_RECURRENT_LAYER(layer)){ CV_ASSERT(k==(n_layers-1));
      CvCNNRecurrentLayer * rnn_layer = (CvCNNRecurrentLayer*)layer;
      n_outputs = rnn_layer->seq_length*rnn_layer->n_output_planes;
    }
    CV_CALL(X[k+1] = cvCreateMat( n_outputs, batch_size, CV_32FC1 ));
    CV_CALL(dE_dX[k+1] = cvCreateMat( batch_size, X[k+1]->rows, CV_32FC1 ));
  }

  for ( n = 1; n <= max_iter; n++ )
  {
    float loss, max_loss = 0;
    int i;
    int nclasses = X[n_layers]->rows;
    CvMat * worst_img_idx = cvCreateMat(batch_size,1,CV_32S);
    int* right_etal_idx = responses->data.i;
    CvMat * etalon = cvCreateMat(batch_size,nclasses,CV_32F);

    // Use the random image
    cvRandArr(&rng,worst_img_idx,CV_RAND_UNI,cvScalar(0),cvScalar(n_images-1));

    // 1) Compute the network output on the <X0_transpose>
    CV_ASSERT(CV_MAT_TYPE(X0_transpose->type)==CV_32F && CV_MAT_TYPE(images->type)==CV_32F);
    for ( k = 0; k < batch_size; k++ ){
      memcpy(X0_transpose->data.fl+img_size*k,
             images->data.fl+images->cols*worst_img_idx->data.i[k],
             sizeof(float)*img_size);
    }
    CV_CALL(cvTranspose( X0_transpose, X[0] ));

    for ( k = 0, layer = first_layer; k < n_layers - 1; k++, layer = layer->next_layer )
#if 1
    { CV_CALL(layer->forward( layer, X[k], X[k+1] )); 
    }
    CV_CALL(layer->forward( layer, X[k], X[k+1] ));
#else
    { CV_CALL(layer->forward( layer, X[k], X[k+1] ));
      if (ICV_IS_CNN_RECURRENT_LAYER(layer)){
        CvCNNLayer * hidden_layer = ((CvCNNRecurrentLayer*)layer)->hidden_layer;
        CvMat * Y = hidden_layer?((CvCNNRecurrentLayer*)hidden_layer)->Y:
                                 ((CvCNNRecurrentLayer*)layer)->Y;
        cvPrintf(stderr,"%.2f ", Y);
        CV_SHOW(Y);
      }// else{icvVisualizeCNNLayer(layer, X[k+1]);}
    }
    CV_CALL(layer->forward( layer, X[k], X[k+1] ));
    // {icvVisualizeCNNLayer(layer, X[k+1]);fprintf(stderr,"\n");}
#endif

    // 2) Compute the gradient
    CvMat etalon_src,etalon_dst;
    if (ICV_IS_CNN_RECURRENT_LAYER(layer)){
      CvCNNLayer * reference_layer = ((CvCNNRecurrentLayer*)layer)->hidden_layer;
      cvCopy(((CvCNNRecurrentLayer*)reference_layer)->Y,dE_dX[n_layers]);
    }else{
      cvTranspose( X[n_layers], dE_dX[n_layers] );
    }
    for ( k = 0; k < batch_size; k++ ){
      cvGetRow( etalons, &etalon_src, responses->data.i[worst_img_idx->data.i[k]] );
      cvGetRow( etalon, &etalon_dst, k );
      cvCopy(&etalon_src, &etalon_dst);
    }
    cvSub( dE_dX[n_layers], etalon, dE_dX[n_layers] );

    // 3) Update weights by the gradient descent
    for ( k = n_layers; k > 0; k--, layer = layer->prev_layer )
#if 1
    { CV_CALL(layer->backward( layer, n + start_iter, X[k-1], dE_dX[k], dE_dX[k-1] ));}
#else
    { CV_CALL(layer->backward( layer, n + start_iter, X[k-1], dE_dX[k], dE_dX[k-1] ));
      CvMat * dE_dX_T = cvCreateMat(dE_dX[k]->cols,dE_dX[k]->rows,CV_32F);
      cvTranspose(dE_dX[k],dE_dX_T);
      icvVisualizeCNNLayer(layer, dE_dX_T);
      cvReleaseMat(&dE_dX_T);
    }fprintf(stderr,"\n");
#endif

#if 1        
    // print progress
    CvMat * mattemp = cvCreateMat(etalon->cols,etalon->rows,CV_MAT_TYPE(etalon->type));
    cvTranspose(etalon, mattemp);
    float trloss = cvNorm(X[n_layers], mattemp)/float(batch_size);
    float top1 = icvEvalAccuracy(X[n_layers], mattemp);
    static double sumloss = 0; sumloss += trloss;
    static double sumacc  = 0; sumacc  += top1;
    if (int(float(n*100)/float(max_iter))<int(float((n+1)*100)/float(max_iter))){
      fprintf(stderr, "%d/%d = %.0f%%,",n+1,max_iter,float(n*100.f)/float(max_iter));
      fprintf(stderr, "sumacc: %.1f%%[%.1f%%], sumloss: %f\n", sumacc/float(n),top1,sumloss/float(n));
    }
    cvReleaseMat(&mattemp);
#endif
    if (etalon){cvReleaseMat(&etalon);etalon=0;}
  }
  if (X0_transpose){cvReleaseMat(&X0_transpose);X0_transpose=0;}
  __END__;

  for ( k = 0; k <= n_layers; k++ ){
    cvReleaseMat( &X[k] );
    cvReleaseMat( &dE_dX[k] );
  }
  cvFree( &X );
  cvFree( &dE_dX );
}

static float icvEvalAccuracy(CvMat * result, CvMat * mattemp)
{
  float top1 = 0;
  int batch_size = result->cols;
  CvMat * sorted = cvCreateMat(result->rows,result->cols,CV_32F);
  CvMat * indices = cvCreateMat(result->rows,result->cols,CV_32S);
  CvMat * indtop1 = cvCreateMat(1,result->cols,CV_32S);
  CvMat * expectedmat = cvCreateMat(1,result->cols,CV_32S);
  CvMat * indtop1true = cvCreateMat(result->rows,result->cols,CV_32S);
  CvMat * indtop1res = cvCreateMat(1,result->cols,CV_8U);
  cvSort(result,sorted,indices,CV_SORT_DESCENDING|CV_SORT_EVERY_COLUMN);
  cvGetRow(indices,indtop1,0);
  cvSort(mattemp,0,indtop1true,CV_SORT_DESCENDING|CV_SORT_EVERY_COLUMN);
  assert( CV_MAT_TYPE(indtop1true->type) == CV_32S && CV_MAT_TYPE(expectedmat->type) == CV_32S );
  for (int ii=0;ii<indtop1true->cols;ii++){
    CV_MAT_ELEM(*expectedmat,int,0,ii)=CV_MAT_ELEM(*indtop1true,int,0,ii); // transpose and convert
  }
  cvCmp(indtop1,expectedmat,indtop1res,CV_CMP_EQ);
  top1=cvSum(indtop1res).val[0]*100.f/float(batch_size)/255.f;
  cvReleaseMat(&sorted);
  cvReleaseMat(&indices);
  cvReleaseMat(&indtop1);
  cvReleaseMat(&expectedmat);
  cvReleaseMat(&indtop1true);
  cvReleaseMat(&indtop1res);
  return top1;
}

/*************************************************************************/
static float icvCNNModelPredict( const CvCNNStatModel* model,
                                 const CvMat* _image,
                                 CvMat* probs )
{
  CvMat** X       = 0;
  float* img_data = 0;
  int n_layers = 0;
  int best_etal_idx = -1;
  int k;

  CV_FUNCNAME("icvCNNModelPredict");
  __BEGIN__;

  CvCNNStatModel* cnn_model = (CvCNNStatModel*)model;
  CvCNNLayer* first_layer, *layer = 0;
  int img_height, img_width, img_size;
  int nclasses, i;
  float loss, min_loss = FLT_MAX;
  float* probs_data;
  CvMat etalon, X0_transpose;
  int nsamples;

  if ( !CV_IS_CNN(model) )
    CV_ERROR( CV_StsBadArg, "Invalid model" );

  nclasses = cnn_model->cls_labels->cols;
  n_layers = cnn_model->network->n_layers;
  first_layer   = cnn_model->network->first_layer;
  img_height = first_layer->input_height;
  img_width  = first_layer->input_width;
  img_size   = img_height*img_width;
  nsamples = _image->rows;
  
#if 0
  cvPreparePredictData( _image, img_size, 0, nclasses, probs, &img_data );
#else
  CV_ASSERT(nsamples==probs->cols);
  CV_ASSERT(_image->cols==img_size && nsamples==_image->rows);
  if (!img_data){img_data = (float*)cvAlloc(img_size*nsamples*sizeof(float));}
  CvMat imghdr = cvMat(_image->rows,_image->cols,CV_32F,img_data);
  cvCopy(_image,&imghdr);
#endif

  // normalize image value range
  double minval, maxval;
  cvMinMaxLoc(&imghdr,&minval,&maxval,0,0);
  cvAddS(&imghdr,cvScalar(-minval),&imghdr);
  cvScale(&imghdr,&imghdr,10./((maxval-minval)*.5f));
  cvAddS(&imghdr,cvScalar(-1.f),&imghdr);

  CV_CALL(X = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
  memset( X, 0, (n_layers+1)*sizeof(CvMat*) );

  CV_CALL(X[0] = cvCreateMat( img_size,nsamples,CV_32FC1 ));
  for ( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer ){
    CV_CALL(X[k+1] = cvCreateMat( layer->n_output_planes*layer->output_height*
                                  layer->output_width, nsamples, CV_32FC1 ));
  }

  X0_transpose = cvMat( nsamples, img_size, CV_32FC1, img_data );
  cvTranspose( &X0_transpose, X[0] );
  for ( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer ) 
#if 1
    {CV_CALL(layer->forward( layer, X[k], X[k+1] ));}
#else
    { // if (k==4){cvScale(X[k],X[k],1./255.);}
      CV_CALL(layer->forward( layer, X[k], X[k+1] ));
      icvVisualizeCNNLayer(layer, X[k+1]);
    }
    fprintf(stderr,"\n");
#endif

  cvCopy(X[n_layers],probs);

  __END__;

  for ( k = 0; k <= n_layers; k++ )
    cvReleaseMat( &X[k] );
  cvFree( &X );
  if ( img_data != _image->data.fl )
    cvFree( &img_data );

  return ((float) ((CvCNNStatModel*)model)->cls_labels->data.i[best_etal_idx]);
}

/****************************************************************************************/
static void icvCNNModelUpdate(
        CvCNNStatModel* _cnn_model, const CvMat* _train_data, int tflag,
        const CvMat* _responses, const CvStatModelParams* _params,
        const CvMat*, const CvMat* _sample_idx,
        const CvMat*, const CvMat* )
{
    const float** out_train_data = 0;
    CvMat* responses             = 0;
    CvMat* cls_labels            = 0;

    CV_FUNCNAME("icvCNNModelUpdate");
    __BEGIN__;

    int n_images, img_size, i;
    CvCNNStatModelParams* params = (CvCNNStatModelParams*)_params;
    CvCNNStatModel* cnn_model = (CvCNNStatModel*)_cnn_model;

    if ( !CV_IS_CNN(cnn_model) ) {
        CV_ERROR( CV_StsBadArg, "Invalid model" );
    }

    // CV_CALL(cvPrepareTrainData( "cvTrainCNNClassifier",
    //     _train_data, tflag, _responses, CV_VAR_CATEGORICAL,
    //     0, _sample_idx, false, &out_train_data,
    //     &n_images, &img_size, &img_size, &responses,
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

    CV_CALL( icvTrainCNNetwork( cnn_model->network, _train_data, responses,
        cnn_model->etalons, params->grad_estim_type, params->max_iter,
        params->start_iter, params->batch_size ));

    __END__;

    cvFree( &out_train_data );
    cvReleaseMat( &responses );
}

/*************************************************************************/
static void icvCNNModelRelease( CvCNNStatModel** cnn_model )
{
    CV_FUNCNAME("icvCNNModelRelease");
    __BEGIN__;

    CvCNNStatModel* cnn;
    if ( !cnn_model )
        CV_ERROR( CV_StsNullPtr, "Null double pointer" );

    cnn = *(CvCNNStatModel**)cnn_model;

    cvReleaseMat( &cnn->cls_labels );
    cvReleaseMat( &cnn->etalons );
    cnn->network->release( &cnn->network );

    cvFree( &cnn );

    __END__;

}

/************************************************************************ \
 *                       Network functions                              *
\************************************************************************/
ML_IMPL CvCNNetwork* cvCreateCNNetwork( CvCNNLayer* first_layer )
{
    CvCNNetwork* network = 0;

    CV_FUNCNAME( "cvCreateCNNetwork" );
    __BEGIN__;

    if ( !ICV_IS_CNN_LAYER(first_layer) )
        CV_ERROR( CV_StsBadArg, "Invalid layer" );

    CV_CALL(network = (CvCNNetwork*)cvAlloc( sizeof(CvCNNetwork) ));
    memset( network, 0, sizeof(CvCNNetwork) );

    network->first_layer    = first_layer;
    network->n_layers  = 1;
    network->release   = icvCNNetworkRelease;
    network->add_layer = icvCNNetworkAddLayer;
    network->get_layer = icvCNNetworkGetLayer;

    __END__;

    if ( cvGetErrStatus() < 0 && network )
        cvFree( &network );

    return network;

}

/***********************************************************************/
static void icvCNNetworkAddLayer( CvCNNetwork* network, CvCNNLayer* layer )
{
  CV_FUNCNAME( "icvCNNetworkAddLayer" );
  __BEGIN__;

  CvCNNLayer* prev_layer;

  if ( network == NULL ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer" );
  }

  // prev_layer = network->first_layer;
  // while ( prev_layer->next_layer ) { prev_layer = prev_layer->next_layer; }
  prev_layer = cvGetCNNLastLayer(network);

  if ( ICV_IS_CNN_FULLCONNECT_LAYER(layer) ){
    if ( layer->n_input_planes != prev_layer->output_width*prev_layer->output_height*
         prev_layer->n_output_planes ) {
      CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
    }
    if ( layer->input_height != 1 || layer->output_height != 1 ||
         layer->input_width != 1  || layer->output_width != 1 ) {
      CV_ERROR( CV_StsBadArg, "Invalid size of the new layer" );
    }
  }else if ( ICV_IS_CNN_CONVOLUTION_LAYER(layer) || ICV_IS_CNN_SUBSAMPLING_LAYER(layer) ){
    if ( prev_layer->n_output_planes != layer->n_input_planes ||
         prev_layer->output_height   != layer->input_height ||
         prev_layer->output_width    != layer->input_width ) {
      CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
    }
  }else if ( ICV_IS_CNN_RECURRENT_LAYER(layer) ) {
    if ( layer->input_height != 1 || layer->output_height != 1 ||
         layer->input_width != 1  || layer->output_width != 1 ) {
      CV_ERROR( CV_StsBadArg, "Invalid size of the new layer" );
    }
  }else if ( ICV_IS_CNN_IMGCROPPING_LAYER(layer) ) {
  }else{
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  layer->prev_layer = prev_layer;
  prev_layer->next_layer = layer;
  network->n_layers++;

  __END__;
}

static CvCNNLayer* icvCNNetworkGetLayer( CvCNNetwork* network, const char * name )
{
  CV_FUNCNAME("icvGetCNNGetLayer");
  CvCNNLayer* first_layer, *layer, *last_layer, *target_layer=0;
  int n_layers, i;
  __BEGIN__;
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;
  first_layer = last_layer = network->first_layer;
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !ICV_IS_CNN_LAYER(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    if (!strcmp(layer->name,name)){target_layer=layer;break;}
    last_layer = layer;
    layer = layer->next_layer;
  }
  __END__;
  return target_layer;
}

CvCNNLayer * cvGetCNNLastLayer(CvCNNetwork * network)
{
  CV_FUNCNAME("icvGetCNNLastLayer");
  CvCNNLayer* first_layer, *layer, *last_layer;
  int n_layers, i;
  __BEGIN__;
  if ( !network ) {
    CV_ERROR( CV_StsNullPtr, "Null <network> pointer. Network must be created by user." ); 
  }
  n_layers = network->n_layers;
  first_layer = last_layer = network->first_layer;
  for ( i = 0, layer = first_layer; i < n_layers && layer; i++ ) {
    if ( !ICV_IS_CNN_LAYER(layer) ) {
      CV_ERROR( CV_StsNullPtr, "Invalid network" );
    }
    last_layer = layer;
    layer = layer->next_layer;
  }
  __END__;
  return last_layer;
}

/*************************************************************************/
static void icvCNNetworkRelease( CvCNNetwork** network_pptr )
{
    CV_FUNCNAME( "icvReleaseCNNetwork" );
    __BEGIN__;

    CvCNNetwork* network = 0;
    CvCNNLayer* layer = 0, *next_layer = 0;
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
static CvCNNLayer* icvCreateCNNLayer( int layer_type, const char * name, int header_size,
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int output_height, int output_width,
    float init_learn_rate, int learn_rate_decrease_type,
    CvCNNLayerRelease release, CvCNNLayerForward forward, CvCNNLayerBackward backward )
{
  CvCNNLayer* layer = 0;

  CV_FUNCNAME("icvCreateCNNLayer");
  __BEGIN__;

  CV_ASSERT( release && forward && backward )
  CV_ASSERT( header_size >= sizeof(CvCNNLayer) )

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
  if ( learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY &&
       learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_SQRT_INV &&
       learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_LOG_INV ) 
  {
    CV_ERROR( CV_StsBadArg, "Invalid type of learning rate dynamics" );
  }

  CV_CALL(layer = (CvCNNLayer*)cvAlloc( header_size ));
  memset( layer, 0, header_size );

  layer->flags = ICV_CNN_LAYER|layer_type;
  CV_ASSERT( ICV_IS_CNN_LAYER(layer) )

  strcpy(layer->name,name);

  layer->n_input_planes = n_input_planes;
  layer->input_height   = input_height;
  layer->input_width    = input_width;

  layer->n_output_planes = n_output_planes;
  layer->output_height   = output_height;
  layer->output_width    = output_width;

  layer->init_learn_rate = init_learn_rate;
  layer->learn_rate_decrease_type = learn_rate_decrease_type;

  layer->release  = release;
  layer->forward  = forward;
  layer->backward = backward;

  __END__;

  if ( cvGetErrStatus() < 0 && layer) { cvFree( &layer ); }

  return layer;
}

void icvVisualizeCNNLayer(CvCNNLayer * layer, const CvMat * Y)
{
  CV_FUNCNAME("icvVisualizeCNNLayer");
  int hh = layer->output_height;
  int ww = layer->output_width;
  int nplanes = layer->n_output_planes;
  int nsamples = Y->cols;
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
      CV_MAT_ELEM(*imgY,float,hh*si+yy,ww*pi+xx)=CV_MAT_ELEM(*Y,float,hh*ww*pi+ww*yy+xx,si);
    }
    }
  }
  }
  fprintf(stderr,"%dx%d,",imgY->cols,imgY->rows);
  CV_SHOW(imgY);
  __END__;
  cvReleaseMat(&imgY);
}

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNConvolutionLayer( const char * name, 
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int K,
    float init_learn_rate, int learn_rate_decrease_type,
    CvMat* connect_mask, CvMat* weights )

{
  CvCNNConvolutionLayer* layer = 0;

  CV_FUNCNAME("cvCreateCNNConvolutionLayer");
  __BEGIN__;

  const int output_height = input_height - K + 1;
  const int output_width = input_width - K + 1;
  fprintf(stderr,"ConvolutionLayer: input (%d@%dx%d), output (%d@%dx%d)\n",
          n_input_planes,input_width,input_height,
          n_output_planes,output_width,output_height);

  if ( K < 1 || init_learn_rate <= 0 || init_learn_rate > 1 ) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  CV_CALL(layer = (CvCNNConvolutionLayer*)icvCreateCNNLayer( ICV_CNN_CONVOLUTION_LAYER, name, 
    sizeof(CvCNNConvolutionLayer), n_input_planes, input_height, input_width,
    n_output_planes, output_height, output_width,
    init_learn_rate, learn_rate_decrease_type,
    icvCNNConvolutionRelease, icvCNNConvolutionForward, icvCNNConvolutionBackward ));

  layer->K = K;
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

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNSubSamplingLayer( const char * name, 
    int n_input_planes, int input_height, int input_width,
    int sub_samp_scale, float a, float s, 
    float init_learn_rate, int learn_rate_decrease_type, CvMat* weights )

{
    CvCNNSubSamplingLayer* layer = 0;

    CV_FUNCNAME("cvCreateCNNSubSamplingLayer");
    __BEGIN__;

    const int output_height   = input_height/sub_samp_scale;
    const int output_width    = input_width/sub_samp_scale;
    const int n_output_planes = n_input_planes;
    fprintf(stderr,"SubSamplingLayer: input (%d@%dx%d), output (%d@%dx%d)\n",
            n_input_planes,input_width,input_height,n_output_planes,output_width,output_height);

    if ( sub_samp_scale < 1 || a <= 0 || s <= 0)
        CV_ERROR( CV_StsBadArg, "Incorrect parameters" );

    CV_CALL(layer = (CvCNNSubSamplingLayer*)icvCreateCNNLayer( ICV_CNN_SUBSAMPLING_LAYER, name, 
        sizeof(CvCNNSubSamplingLayer), n_input_planes, input_height, input_width,
        n_output_planes, output_height, output_width,
        init_learn_rate, learn_rate_decrease_type,
        icvCNNSubSamplingRelease, icvCNNSubSamplingForward, icvCNNSubSamplingBackward ));

    layer->sub_samp_scale  = sub_samp_scale;
    layer->a               = a;
    layer->s               = s;
    layer->mask = 0;

    CV_CALL(layer->sumX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));
    CV_CALL(layer->exp2ssumWX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));
    
    cvZero( layer->sumX );
    cvZero( layer->exp2ssumWX );

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
        cvReleaseMat( &layer->exp2ssumWX );
        cvFree( &layer );
    }

    return (CvCNNLayer*)layer;
}

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNFullConnectLayer( const char * name, 
    int n_inputs, int n_outputs, float a, float s, 
    float init_learn_rate, int learn_rate_decrease_type, int activation_type, CvMat* weights )
{
  CvCNNFullConnectLayer* layer = 0;

  CV_FUNCNAME("cvCreateCNNFullConnectLayer");
  __BEGIN__;

  if ( a <= 0 || s <= 0 || init_learn_rate <= 0) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  fprintf(stderr,"FullConnectLayer: input (%d), output (%d)\n",
          n_inputs,n_outputs);
  
  CV_CALL(layer = (CvCNNFullConnectLayer*)icvCreateCNNLayer( ICV_CNN_FULLCONNECT_LAYER, name, 
      sizeof(CvCNNFullConnectLayer), n_inputs, 1, 1, n_outputs, 1, 1,
      init_learn_rate, learn_rate_decrease_type,
      icvCNNFullConnectRelease, icvCNNFullConnectForward, icvCNNFullConnectBackward ));

  layer->a = a;
  layer->s = s;
  layer->exp2ssumWX = 0;
  layer->activation_type = activation_type;//CV_CNN_HYPERBOLIC;

  CV_CALL(layer->weights = cvCreateMat( n_outputs, n_inputs+1, CV_32FC1 ));
  if ( weights ){
    if ( !ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ) ) {
      CV_ERROR( CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1" );
    }
    if ( !CV_ARE_SIZES_EQ( weights, layer->weights ) ) {
      CV_ERROR( CV_StsBadSize, "Invalid size of initial weights matrix" );
    }
    CV_CALL(cvCopy( weights, layer->weights ));
  } else {
    CvRNG rng = cvRNG( 0xFFFFFFFF );
    cvRandArr( &rng, layer->weights, CV_RAND_UNI, 
               cvScalar(-1.f/sqrt(n_inputs)), cvScalar(1.f/sqrt(n_inputs)) );
    // initialize bias to zero
    for (int ii=0;ii<n_outputs;ii++){ CV_MAT_ELEM(*layer->weights,float,ii,n_inputs)=0; }
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvReleaseMat( &layer->exp2ssumWX );
    cvReleaseMat( &layer->weights );
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

/*************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNRecurrentLayer( const char * name, 
    const CvCNNLayer * hidden_layer, 
    int n_inputs, int n_outputs, int n_hiddens, int seq_length, int time_index, 
    float init_learn_rate, int update_rule, int activation_type, 
    CvMat * Wxh, CvMat * Whh, CvMat * Why )
{
  CvCNNRecurrentLayer* layer = 0;

  CV_FUNCNAME("cvCreateCNNRecurrentLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) { CV_ERROR( CV_StsBadArg, "Incorrect parameters" ); }

  fprintf(stderr,"RecurrentLayer: input (%d), hidden (%d), output (%d), seq_length (%d), time_index (%d)\n",
          n_inputs, n_hiddens, n_outputs, seq_length, time_index);
  
  CV_CALL(layer = (CvCNNRecurrentLayer*)icvCreateCNNLayer( ICV_CNN_RECURRENT_LAYER, name, 
      sizeof(CvCNNRecurrentLayer), n_inputs, 1, 1, n_outputs, 1, 1,
      init_learn_rate, update_rule,
      icvCNNRecurrentRelease, icvCNNRecurrentForward, icvCNNRecurrentBackward ));

  layer->hidden_layer = (CvCNNLayer*)hidden_layer;
  layer->weights = 0; // we don't use this !
  layer->time_index = time_index;
  layer->seq_length = seq_length;
  layer->n_hiddens = n_hiddens;
  layer->H = 0;
  layer->Y = 0;
  layer->P = 0;
  layer->loss = 0;
  layer->Wxh = 0;
  layer->Whh = 0;
  layer->Why = 0;
  layer->activation_type = activation_type;

  int n_hiddens = layer->n_hiddens;
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

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvReleaseMat( &layer->Wxh );
    cvReleaseMat( &layer->Whh );
    cvReleaseMat( &layer->Why );
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

CvCNNLayer * cvCreateCNNImgCroppingLayer( const char * name, const CvCNNLayer * _image_layer,
    int n_output_planes, int output_height, int output_width, int time_index,
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
  CV_ASSERT(ICV_IS_CNN_INPUTDATA_LAYER(_image_layer));

  fprintf(stderr,"ImgCroppingLayer: "
          "input (%d@%dx%d), output (%d@%dx%d), seq_length: %d, time_index: %d\n",
          n_inputs,input_layer->input_height,input_layer->input_width,
          n_outputs,output_height,output_width,input_layer->seq_length,time_index);
  
  CV_CALL(layer = (CvCNNImgCroppingLayer*)icvCreateCNNLayer( ICV_CNN_IMGCROPPING_LAYER, name, 
      sizeof(CvCNNImgCroppingLayer), 
      n_inputs, input_layer->input_height, input_layer->input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNImgCroppingRelease, icvCNNImgCroppingForward, icvCNNImgCroppingBackward ));

  layer->input_layer = (CvCNNLayer*)_image_layer;
  layer->time_index = time_index;

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

CvCNNLayer * cvCreateCNNInputDataLayer( const char * name, 
    int n_input_planes, int input_height, int input_width, int seq_length,
    float init_learn_rate, int update_rule)
{
  CvCNNInputDataLayer* layer = 0;
  const int n_inputs = n_input_planes;
  const int n_outputs = n_input_planes;
  const int output_width = input_width;
  const int output_height = input_height;

  CV_FUNCNAME("cvCreateCNNInputDataLayer");
  __BEGIN__;

  if ( init_learn_rate <= 0) {
    CV_ERROR( CV_StsBadArg, "Incorrect parameters" );
  }

  fprintf(stderr,"InputDataLayer: input (%d@%dx%d), output (%d@%dx%d), seq_length: %d\n",
          n_inputs,input_height,input_width,
          n_outputs,output_height,output_width,seq_length);
  
  CV_CALL(layer = (CvCNNInputDataLayer*)icvCreateCNNLayer( ICV_CNN_INPUTDATA_LAYER, name, 
      sizeof(CvCNNInputDataLayer), n_inputs, input_height, input_width, 
      n_outputs, output_height, output_width, init_learn_rate, update_rule,
      icvCNNInputDataRelease, icvCNNInputDataForward, icvCNNInputDataBackward ));

  layer->seq_length = seq_length;

  __END__;

  if ( cvGetErrStatus() < 0 && layer ){
    cvFree( &layer );
  }

  return (CvCNNLayer*)layer;
}

/*************************************************************************\
 *                 Layer FORWARD functions                               *
\*************************************************************************/
static void 
icvCNNConvolutionForward( CvCNNLayer* _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNConvolutionForward");

  if ( !ICV_IS_CNN_CONVOLUTION_LAYER(_layer) )
    CV_ERROR( CV_StsBadArg, "Invalid layer" );

  {__BEGIN__;

    const CvCNNConvolutionLayer* layer = (CvCNNConvolutionLayer*) _layer;

    const int K = layer->K;
    const int n_weights_for_Yplane = K*K + 1;

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
      float * wptr = layer->weights->data.fl+n_weights_for_Yplane*no;
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
  }__END__;
}

/**************************************************************************/
static void 
icvCNNSubSamplingForward( CvCNNLayer* _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNSubSamplingForward");

  if ( !ICV_IS_CNN_SUBSAMPLING_LAYER(_layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  {__BEGIN__;

  CvCNNSubSamplingLayer* layer = (CvCNNSubSamplingLayer*) _layer;

  const int stride_size = layer->sub_samp_scale;
  const int nsamples = X->cols; // batch size for training
  const int nplanes = layer->n_input_planes;
  const int Xheight = layer->input_height;
  const int Xwidth  = layer->input_width ;
  const int Xsize   = Xwidth*Xheight;
  const int Yheight = layer->output_height;
  const int Ywidth  = layer->output_width;
  const int Ysize   = Ywidth*Yheight;
  const int batch_size = X->cols;

  int xx, yy, ni, kx, ky;
  float* sumX_data = 0, *w = 0;
  CvMat sumX_sub_col, exp2ssumWX_sub_col;

  CV_ASSERT(X->rows == nplanes*Xsize && X->cols == nsamples);
  CV_ASSERT(Y->cols == nsamples);
  CV_ASSERT((layer->exp2ssumWX->cols == 1) &&
            (layer->exp2ssumWX->rows == nplanes*Ysize));

  CV_CALL(layer->mask = cvCreateMat(Ysize*nplanes, batch_size, CV_32S));
  
  // update inner variable used in back-propagation
  cvZero( layer->sumX );
  cvZero( layer->exp2ssumWX );
  cvZero( layer->mask );
  
  int * mptr = layer->mask->data.i;

  CV_ASSERT(Xheight==Yheight*stride_size && Xwidth ==Ywidth *stride_size);
  CV_ASSERT(Y->rows==layer->mask->rows && Y->cols==layer->mask->cols);
  CV_ASSERT(CV_MAT_TYPE(layer->mask->type)==CV_32S);

  CvMat * Xt = cvCreateMat(X->cols,X->rows,CV_32F); cvTranspose(X,Xt);
  CvMat * Yt = cvCreateMat(Y->cols,Y->rows,CV_32F); cvTranspose(Y,Yt);
  for ( int si = 0; si < nsamples; si++ ){
  float * xptr = Xt->data.fl+Xsize*nplanes*si;
  float * yptr = Yt->data.fl+Ysize*nplanes*si;
  for ( ni = 0; ni < nplanes; ni++ ){
    for ( yy = 0; yy < Yheight; yy++ ){
    for ( xx = 0; xx < Ywidth; xx++ ){
      float maxval = *xptr;
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
  cvTranspose(Yt,Y);
  cvReleaseMat(&Xt);
  cvReleaseMat(&Yt);

  }__END__;
}

/****************************************************************************************/
static void icvCNNFullConnectForward( CvCNNLayer* _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNFullConnectForward");

  if ( !ICV_IS_CNN_FULLCONNECT_LAYER(_layer) ) {
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  {__BEGIN__;

  CvCNNFullConnectLayer * layer = (CvCNNFullConnectLayer*)_layer;
  CvMat * weights = layer->weights;
  CvMat sub_weights, biascol;
  int n_outputs = Y->rows;
  int batch_size = X->cols;

  CV_ASSERT(X->cols == batch_size && X->rows == layer->n_input_planes);
  CV_ASSERT(Y->cols == batch_size && Y->rows == layer->n_output_planes);

  // bias on last column vector
  CvRect roi = cvRect(0, 0, weights->cols-1, weights->rows );
  CV_CALL(cvGetSubRect( weights, &sub_weights, roi));
  CV_CALL(cvGetCol( weights, &biascol, weights->cols-1));
  CV_ASSERT(CV_MAT_TYPE(biascol.type)==CV_32F);
  CvMat * bias = cvCreateMat(biascol.rows,batch_size,CV_32F);
  cvRepeat(&biascol,bias);
  
  CV_CALL(layer->exp2ssumWX = cvCreateMat( n_outputs, batch_size, CV_32FC1 ));
  cvZero( layer->exp2ssumWX );

  // update inner variables used in back-propagation
  CV_CALL(cvGEMM( &sub_weights, X, 1, bias, 1, layer->exp2ssumWX ));
  if (layer->activation_type==CV_CNN_NONE){
    // do nothing
  }else if (layer->activation_type==CV_CNN_HYPERBOLIC){
    CV_CALL(cvTanh( layer->exp2ssumWX, Y ));
  }else if (layer->activation_type==CV_CNN_LOGISTIC){
    CV_CALL(cvSigmoid( layer->exp2ssumWX, Y ));
  }else if (layer->activation_type==CV_CNN_RELU){
    CV_CALL(cvReLU( layer->exp2ssumWX, Y ));
  }else{assert(false);}

  //// check numerical stability
  // CV_CALL(cvMinS( layer->exp2ssumWX, FLT_MAX, layer->exp2ssumWX ));
  // for ( int ii = 0; ii < layer->exp2ssumWX->rows; ii++ ){
  //   CV_ASSERT ( layer->exp2ssumWX->data.fl[ii] != FLT_MAX );
  // }

  //// compute the output variable
  // CvMat * sum = cvCreateMat(1,batch_size,CV_32F);
  // CvMat * sumrep = cvCreateMat(n_outputs,batch_size,CV_32F);
  // cvReduce(layer->exp2ssumWX,sum,-1,CV_REDUCE_SUM);cvRepeat(sum,sumrep);
  // cvDiv(layer->exp2ssumWX,sumrep,Y);
  // cvReleaseMat(&sum);
  // cvReleaseMat(&sumrep);

  if (bias){cvReleaseMat(&bias);bias=0;}

  }__END__;
}

static void icvCNNImgCroppingForward( CvCNNLayer * _layer, const CvMat* X, CvMat* Y )
{
  CV_FUNCNAME("icvCNNImgCroppingForward");
  if ( !ICV_IS_CNN_IMGCROPPING_LAYER(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  const CvCNNImgCroppingLayer * layer = (CvCNNImgCroppingLayer*)_layer;
  const CvCNNInputDataLayer * input_layer = (CvCNNInputDataLayer*)layer->input_layer;
  const int time_index = layer->time_index;
  const int seq_length = input_layer->seq_length;
  const int n_outputs = Y->rows;
  CV_ASSERT(Y->rows==layer->n_output_planes*layer->output_height*layer->output_width);
  CV_ASSERT(X->cols==Y->cols); // batch_size
  CvMat * input_data = input_layer->input_data;
  CV_ASSERT(CV_MAT_TYPE(input_data->type)==CV_32F);
  CvMat input_data_submat;
  CvMat input_data_hdr = cvMat(n_outputs,seq_length,CV_32F,input_data->data.ptr);
  cvGetCol(&input_data_hdr,&input_data_submat,time_index);
  cvCopy(&input_data_submat,Y);
  __END__;
}

/****************************************************************************************/
static void icvCNNRecurrentForward( CvCNNLayer* _layer, const CvMat* X, CvMat * Y) 
{
  CV_FUNCNAME("icvCNNRecurrentForward");
  if ( !ICV_IS_CNN_RECURRENT_LAYER(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;

  CvCNNRecurrentLayer * layer = (CvCNNRecurrentLayer*)_layer;
  CvCNNRecurrentLayer * hidden_layer = (CvCNNRecurrentLayer*)layer->hidden_layer;
  CvMat * Wxh = layer->Wxh;
  CvMat * Whh = layer->Whh;
  CvMat * Why = layer->Why;
  CvMat Wxh_submat, Whh_submat, hbiascol, Why_submat, ybiascol;
  int time_index = layer->time_index;
  int seq_length = layer->seq_length;
  int n_inputs = layer->n_input_planes;//Y->rows;
  int n_outputs = layer->n_output_planes;//Y->rows;
  int n_hiddens = layer->n_hiddens;
  int batch_size = X->cols;
  CvMat * WX = 0, * WH = 0, * H_prev = 0, * H_curr = 0;

  CV_ASSERT(X->cols == batch_size && X->rows == layer->n_input_planes);
  // CV_ASSERT(X_next->cols == batch_size && X_next->rows == layer->n_input_planes);

  // memory allocation
  if (!hidden_layer){
    layer->H = cvCreateMat( n_hiddens * batch_size, seq_length, CV_32F ); cvZero(layer->H);
    layer->Y = cvCreateMat( n_outputs * batch_size, seq_length, CV_32F ); cvZero(layer->Y);
    layer->P = cvCreateMat( n_outputs * batch_size, seq_length, CV_32F ); cvZero(layer->P);
  }
  CvMat * H = hidden_layer?hidden_layer->H:layer->H;
  CV_CALL(WX = cvCreateMat( n_hiddens, batch_size, CV_32F ));
  CV_CALL(WH = cvCreateMat( n_hiddens, batch_size, CV_32F ));
  CV_CALL(H_prev = cvCreateMat( n_hiddens * batch_size, 1, CV_32F ));
  CV_CALL(H_curr = cvCreateMat( n_hiddens * batch_size, 1, CV_32F ));
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
  CvMat H_prev_hdr, H_curr_hdr;
  if (layer->time_index==0){ cvZero(H_prev); }else{
    cvGetCol(H,&H_prev_hdr,layer->time_index-1); cvCopy(&H_prev_hdr,H_prev);
  }
  CvMat H_prev_reshaped = cvMat(n_hiddens, batch_size, CV_32F, H_prev->data.ptr);
  CvMat H_curr_reshaped = cvMat(n_hiddens, batch_size, CV_32F, H_curr->data.ptr);

  // H_t1 = Wxh * X_t0 + ( Whh * H_t0 + bh )
  CV_CALL(cvGEMM( Wxh, X, 1, 0, 1, WX ));
  CV_CALL(cvGEMM( &Whh_submat, &H_prev_reshaped, 1, hbias, 1, WH ));
  cvAdd(WX,WH,&H_curr_reshaped);
  
  // activation for hidden states, relu and tahn is preferred
  if (layer->activation_type==CV_CNN_NONE){
  }else if (layer->activation_type==CV_CNN_HYPERBOLIC){
    CV_CALL(cvTanh( H_curr, H_curr ));
  }else if (layer->activation_type==CV_CNN_LOGISTIC){
    CV_CALL(cvSigmoid( H_curr, H_curr ));
  }else if (layer->activation_type==CV_CNN_RELU){
    CV_CALL(cvReLU( H_curr, H_curr ));
  }else{assert(false);}

  CvMat Y_curr_hdr,P_curr_hdr; CV_ASSERT(batch_size==1);
  cvGetCol(H,&H_curr_hdr,layer->time_index); cvCopy(H_curr,&H_curr_hdr);
  cvGetCol(hidden_layer?hidden_layer->Y:layer->Y,&Y_curr_hdr,layer->time_index); 
  cvGetCol(hidden_layer?hidden_layer->P:layer->P,&P_curr_hdr,layer->time_index); 

  // Y = Why * H + by
  CV_CALL(cvGEMM( &Why_submat, &H_curr_reshaped, 1, ybias, 1, &Y_curr_hdr ));
  if (layer->next_layer) {cvCopy(&Y_curr_hdr,Y); }else{ cvTranspose(hidden_layer->Y,Y); }

  // size correct ???? and compute loss ???
  CvMat * sumP = cvCreateMat(P_curr_hdr.rows,P_curr_hdr.cols,CV_32F); 
  cvExp(&Y_curr_hdr,&P_curr_hdr);
  cvReduce(&P_curr_hdr,sumP,-1,CV_REDUCE_SUM);
  cvReleaseMat(&sumP);
  
  if (WX){cvReleaseMat(&WX);WX=0;}
  if (WH){cvReleaseMat(&WH);WH=0;}
  if (H_prev){cvReleaseMat(&H_prev);H_prev=0;}
  if (H_curr){cvReleaseMat(&H_curr);H_curr=0;}
  if (hbias){cvReleaseMat(&hbias);hbias=0;}
  if (ybias){cvReleaseMat(&ybias);ybias=0;}

  __END__;
}

static void icvCNNInputDataForward( CvCNNLayer * _layer, const CvMat* X, CvMat*  )
{
  CV_FUNCNAME("icvCNNInputDataForward");
  if ( !ICV_IS_CNN_INPUTDATA_LAYER(_layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }
  __BEGIN__;
  CvCNNInputDataLayer * layer = (CvCNNInputDataLayer*)_layer;
  if (!layer->input_data){
    layer->input_data = cvCloneMat(X);
  }

  __END__;
}

/*************************************************************************\
*                           Layer BACKWARD functions                      *
\*************************************************************************/

/* <dE_dY>, <dE_dX> should be row-vectors.
   Function computes partial derivatives <dE_dX>
   of the loss function with respect to the planes components
   of the previous layer (X).
   It is a basic function for back propagation method.
   Input parameter <dE_dY> is the partial derivative of the
   loss function with respect to the planes components
   of the current layer. */
static void icvCNNConvolutionBackward(
    CvCNNLayer* _layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX )
{
  CvMat* dY_dX = 0;
  CvMat* dY_dW = 0;
  CvMat* dE_dW = 0;

  CV_FUNCNAME("icvCNNConvolutionBackward");

  if ( !ICV_IS_CNN_CONVOLUTION_LAYER(_layer) )
    CV_ERROR( CV_StsBadArg, "Invalid layer" );

  {__BEGIN__;

  const CvCNNConvolutionLayer* layer = (CvCNNConvolutionLayer*) _layer;
  
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

  // int no, ni, yy, xx, ky, kx;
  // int xloc = 0, yloc = 0;
  // float *X_plane = 0, *w = 0;
  // CvMat* weights = layer->weights;

  CV_ASSERT( t >= 1 );
  CV_ASSERT( n_Y_planes == layer->weights->rows );

  dY_dX = cvCreateMat( n_Y_planes*Y_plane_size, X->rows, CV_32FC1 );
  dY_dW = cvCreateMat( dY_dX->rows, layer->weights->cols*layer->weights->rows, CV_32FC1 );
  dE_dW = cvCreateMat( 1, dY_dW->cols, CV_32FC1 );

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
    float * wptr = layer->weights->data.fl + noKK;
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
    if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_LOG_INV ) {
      eta = -layer->init_learn_rate/logf(1+(float)t);
    } else if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV ) {
      eta = -layer->init_learn_rate/sqrtf((float)t);
    } else {
      eta = -layer->init_learn_rate/(float)t;
    }
    cvReshape( dE_dW, &dE_dW_mat, 0, layer->weights->rows );
    cvScaleAdd( &dE_dW_mat, cvRealScalar(eta), layer->weights, layer->weights );
  }

  }__END__;

  cvReleaseMat( &dY_dX );
  cvReleaseMat( &dY_dW );
  cvReleaseMat( &dE_dW );
}

/****************************************************************************************/
static void icvCNNSubSamplingBackward(
    CvCNNLayer* _layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX )
{
  // derivative of activation function
  CvMat* dY_dX_elems = 0; // elements of matrix dY_dX
  CvMat* dY_dW_elems = 0; // elements of matrix dY_dW
  CvMat* dE_dW = 0;

  CV_FUNCNAME("icvCNNSubSamplingBackward");

  if ( !ICV_IS_CNN_SUBSAMPLING_LAYER(_layer) ) {
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

  int nplanes = layer->n_output_planes;
  int nsamples = X->cols;
  int stride_size = layer->sub_samp_scale;

  CvMat * maskT = cvCreateMat(dE_dY->rows,dE_dY->cols,CV_32S);
  cvTranspose(layer->mask,maskT);
  cvZero(dE_dX);
  for ( int si = 0; si < nsamples; si++ ){
  float * dxptr = dE_dX->data.fl+dE_dX->cols*si;
  float * dyptr = dE_dY->data.fl+dE_dY->cols*si;
  int * mptr = maskT->data.i;
  for ( int ni = 0; ni < nplanes; ni++ ){
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
  cvReleaseMat(&maskT);
  
  if (layer->mask){cvReleaseMat(&layer->mask);layer->mask=0;}
  __END__;

  if (dY_dX_elems){cvReleaseMat( &dY_dX_elems );dY_dX_elems=0;}
  if (dY_dW_elems){cvReleaseMat( &dY_dW_elems );dY_dW_elems=0;}
  if (dE_dW      ){cvReleaseMat( &dE_dW       );dE_dW      =0;}
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
static void icvCNNFullConnectBackward( CvCNNLayer* _layer,
                                    int t,
                                    const CvMat* X,
                                    const CvMat* dE_dY,
                                    CvMat* dE_dX )
{
  CvMat* dE_dY_afder = 0;
  CvMat* dE_dW = 0;

  CV_FUNCNAME( "icvCNNFullConnectBackward" );

  if ( !ICV_IS_CNN_FULLCONNECT_LAYER(_layer) ) {
      CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  {__BEGIN__;

  CvCNNFullConnectLayer* layer = (CvCNNFullConnectLayer*)_layer;
  const int n_outputs = layer->n_output_planes;
  const int n_inputs  = layer->n_input_planes;

  int i;
  CvMat* weights = layer->weights;
  CvMat sub_weights, Xtemplate, exp2ssumWXrow;
  int batch_size = X->cols;
  CvMat * dE_dY_T = cvCreateMat(n_outputs, batch_size, CV_32F);

  CV_ASSERT(X->cols == batch_size && X->rows == n_inputs);
  CV_ASSERT(dE_dY->rows == batch_size && dE_dY->cols == n_outputs );
  CV_ASSERT(dE_dX->rows == batch_size && dE_dX->cols == n_inputs );

  CV_CALL(dE_dY_afder = cvCreateMat( n_outputs, batch_size, CV_32FC1 ));
  CV_CALL(dE_dW = cvCreateMat( 1, weights->rows*weights->cols, CV_32FC1 ));

  // compute (tanh'(WX))*dE_dY
  if (layer->activation_type==CV_CNN_NONE){
  }else if (layer->activation_type==CV_CNN_HYPERBOLIC){
    // cvTanh(layer->exp2ssumWX,dE_dY_afder);
    // cvPow(dE_dY_afder,dE_dY_afder,2.);
    // cvSubRS(dE_dY_afder,cvScalar(1),dE_dY_afder);
    cvTanhDer(layer->exp2ssumWX,dE_dY_afder);
  }else if (layer->activation_type==CV_CNN_LOGISTIC){
    // cvSigmoid(layer->exp2ssumWX,dE_dY_afder);
    // CvMat * dE_dY_clone = cvCloneMat(dE_dY_afder);
    // cvSubRS(dE_dY_afder,cvScalar(1),dE_dY_afder);
    // cvMul(dE_dY_clone,dE_dY_afder,dE_dY_afder);
    // cvReleaseMat(&dE_dY_clone);
    cvSigmoidDer(layer->exp2ssumWX,dE_dY_afder);
  }else if (layer->activation_type==CV_CNN_RELU){
    // CvMat * dE_dY_mask = cvCreateMat(dE_dY_afder->rows,dE_dY_afder->cols,CV_8U);
    // cvCmpS(layer->exp2ssumWX,0,dE_dY_mask,CV_CMP_GT);
    // cvScale(dE_dY_mask,dE_dY_afder,1./255.f);
    // cvReleaseMat(&dE_dY_mask);
    cvReLUDer(layer->exp2ssumWX,dE_dY_afder);
  }else{assert(false);}
  cvTranspose(dE_dY,dE_dY_T);
  cvMul(dE_dY_afder,dE_dY_T,dE_dY_afder);

  // compute dE_dX=dE_dY_afder*W
  CV_CALL(cvGetCols( weights, &sub_weights, 0, weights->cols-1 ));
  CV_CALL(cvGEMM( dE_dY_afder, &sub_weights, 1, 0, 1, dE_dX, CV_GEMM_A_T));
  
  // compute dE_dW=dE_dY*X
  cvZero(dE_dW);
  CV_ASSERT( dE_dY->rows == batch_size && dE_dY->cols == n_outputs );
  CvMat dE_dW_hdr = cvMat(n_outputs,n_inputs+1,CV_32F,dE_dW->data.fl);
  CvMat * Xcol = cvCreateMat(X->rows+1,X->cols,CV_32F); 
  CvMat Xcol_submat;
  cvSet(Xcol,cvScalar(1)); // all ones on last row
  cvGetRows(Xcol,&Xcol_submat,0,X->rows);
  cvCopy(X,&Xcol_submat);
  cvGEMM(dE_dY,Xcol,1,0,1,&dE_dW_hdr,CV_GEMM_A_T|CV_GEMM_B_T);
  cvReleaseMat(&Xcol);

  // 2) update weights
  {
    float eta;
    if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_LOG_INV ){
      eta = -layer->init_learn_rate/logf(1+(float)t);
    }else if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV ){
      eta = -layer->init_learn_rate/sqrtf((float)t);
    }else{
      eta = -layer->init_learn_rate/(float)t;
    }
    cvScaleAdd( &dE_dW_hdr, cvRealScalar(eta), weights, weights );
  }
  
  cvReleaseMat(&dE_dY_T);
  if (layer->exp2ssumWX){ cvReleaseMat(&layer->exp2ssumWX);layer->exp2ssumWX=0; }
  }__END__;

  cvReleaseMat( &dE_dY_afder );
  cvReleaseMat( &dE_dW );
}

static void icvCNNImgCroppingBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX ){}

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
static void icvCNNRecurrentBackward( CvCNNLayer* _layer,
                                    int t,
                                    const CvMat* X,
                                    const CvMat* dE_dY,
                                    CvMat* dE_dX )
{
  CvMat* dE_dY_afder = 0;
  CvMat* dE_dW = 0;

  CV_FUNCNAME( "icvCNNRecurrentBackward" );

  if ( !ICV_IS_CNN_RECURRENT_LAYER(_layer) ) {
      CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  {__BEGIN__;

  CvCNNRecurrentLayer* layer = (CvCNNRecurrentLayer*)_layer;
  const int n_outputs = layer->n_output_planes;
  const int n_inputs  = layer->n_input_planes;

#if 0
  int i;
  CvMat* weights = layer->weights;
  CvMat sub_weights, Xtemplate, exp2ssumWXrow;
  int batch_size = X->cols;
  CvMat * dE_dY_T = cvCreateMat(n_outputs, batch_size, CV_32F);
  CvMat * WX = 0;

  CV_ASSERT(X->cols == batch_size && X->rows == n_inputs);
  CV_ASSERT(dE_dY->rows == batch_size && dE_dY->cols == n_outputs );
  CV_ASSERT(dE_dX->rows == batch_size && dE_dX->cols == n_inputs );

  CV_CALL(dE_dY_afder = cvCreateMat( n_outputs, batch_size, CV_32FC1 ));
  CV_CALL(dE_dW = cvCreateMat( 1, weights->rows*weights->cols, CV_32FC1 ));

  // compute (tanh'(WX))*dE_dY
  if (layer->activation_type==CV_CNN_NONE){
  }else if (layer->activation_type==CV_CNN_HYPERBOLIC){
    cvTanhDer(WX,dE_dY_afder);
  }else if (layer->activation_type==CV_CNN_LOGISTIC){
    cvSigmoidDer(WX,dE_dY_afder);
  }else if (layer->activation_type==CV_CNN_RELU){
    cvReLUDer(WX,dE_dY_afder);
  }else{assert(false);}
  cvTranspose(dE_dY,dE_dY_T);
  cvMul(dE_dY_afder,dE_dY_T,dE_dY_afder);

  // compute dE_dX=dE_dY_afder*W
  CV_CALL(cvGetCols( weights, &sub_weights, 0, weights->cols-1 ));
  CV_CALL(cvGEMM( dE_dY_afder, &sub_weights, 1, 0, 1, dE_dX, CV_GEMM_A_T));
  
  // compute dE_dW=dE_dY*X
  cvZero(dE_dW);
  CV_ASSERT( dE_dY->rows == batch_size && dE_dY->cols == n_outputs );
  CvMat dE_dW_hdr = cvMat(n_outputs,n_inputs+1,CV_32F,dE_dW->data.fl);
  CvMat * Xcol = cvCreateMat(X->rows+1,X->cols,CV_32F); 
  CvMat Xcol_submat;
  cvSet(Xcol,cvScalar(1)); // all ones on last row
  cvGetRows(Xcol,&Xcol_submat,0,X->rows);
  cvCopy(X,&Xcol_submat);
  cvGEMM(dE_dY,Xcol,1,0,1,&dE_dW_hdr,CV_GEMM_A_T|CV_GEMM_B_T);
  cvReleaseMat(&Xcol);

  // 2) update weights
  {
    float eta;
    if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_LOG_INV ){
      eta = -layer->init_learn_rate/logf(1+(float)t);
    }else if ( layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV ){
      eta = -layer->init_learn_rate/sqrtf((float)t);
    }else{
      eta = -layer->init_learn_rate/(float)t;
    }
    cvScaleAdd( &dE_dW_hdr, cvRealScalar(eta), weights, weights );
  }
  
  cvReleaseMat(&dE_dY_T);
  if (WX){ cvReleaseMat(&WX);WX=0; }
#endif
  }__END__;

  cvReleaseMat( &dE_dY_afder );
  cvReleaseMat( &dE_dW );
}

static void icvCNNInputDataBackward( CvCNNLayer* layer, int t, const CvMat*, const CvMat* dE_dY, CvMat* dE_dX ){}

/*************************************************************************\
 *                           Utility functions                           *
\*************************************************************************/
void cvTanh(CvMat * src, CvMat * dst)
{
  CV_FUNCNAME("cvTanh");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows);
  CV_ASSERT(src->cols==dst->cols);
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
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
  // cvTanh(src,dst);
  // cvPow(dst,dst,2.);
  // cvSubRS(dst,cvScalar(1),dst);
  CV_FUNCNAME("cvTanh");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = 1.f-pow(tanh(srcptr[ii]),2);
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}

void cvSigmoid(CvMat * src, CvMat * dst)
{
  CV_FUNCNAME("cvSigmoid");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows);
  CV_ASSERT(src->cols==dst->cols);
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      dstptr[ii] = 1.f/(1.f+exp(-srcptr[ii]));
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}

void cvSigmoidDer(CvMat * src, CvMat * dst) {
  cvSigmoid(src,dst);
  CvMat * tmp = cvCloneMat(dst);
  cvSubRS(dst,cvScalar(1),dst);
  cvMul(tmp,dst,dst);
  cvReleaseMat(&tmp);
}

void cvReLUDer(CvMat * src, CvMat * dst) {
  CV_FUNCNAME("cvReLUDer");
  int ii,elemsize=src->rows*src->cols;
  __CV_BEGIN__
  {
  CV_ASSERT(src->rows==dst->rows && src->cols==dst->cols);
  if (CV_MAT_TYPE(src->type)==CV_32F){
    float * srcptr = src->data.fl;
    float * dstptr = dst->data.fl;
    for (ii=0;ii<elemsize;ii++){
      if (srcptr[ii]>0){dstptr[ii] = 1.f;}else {dstptr[ii] = 0.f;}
    }
  }else{
    CV_ERROR(CV_StsBadArg,"Unsupported data type");
  }
  }
  __CV_END__
}
  
/*************************************************************************\
*                           Layer RELEASE functions                       *
\*************************************************************************/
static void icvCNNConvolutionRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNConvolutionRelease");
  __BEGIN__;

  CvCNNConvolutionLayer* layer = 0;

  if ( !p_layer )
      CV_ERROR( CV_StsNullPtr, "Null double pointer" );

  layer = *(CvCNNConvolutionLayer**)p_layer;

  if ( !layer )
      return;
  if ( !ICV_IS_CNN_CONVOLUTION_LAYER(layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  cvReleaseMat( &layer->weights );
  cvReleaseMat( &layer->connect_mask );
  cvFree( p_layer );

  __END__;
}

/****************************************************************************************/
static void icvCNNSubSamplingRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNSubSamplingRelease");
  __BEGIN__;

  CvCNNSubSamplingLayer* layer = 0;

  if ( !p_layer )
      CV_ERROR( CV_StsNullPtr, "Null double pointer" );

  layer = *(CvCNNSubSamplingLayer**)p_layer;

  if ( !layer )
      return;
  if ( !ICV_IS_CNN_SUBSAMPLING_LAYER(layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  if (layer->mask      ){cvReleaseMat( &layer->mask       ); layer->mask      =0;}
  if (layer->exp2ssumWX){cvReleaseMat( &layer->exp2ssumWX ); layer->exp2ssumWX=0;}
  if (layer->weights   ){cvReleaseMat( &layer->weights    ); layer->weights   =0;}
  cvFree( p_layer );

  __END__;
}

/****************************************************************************************/
static void icvCNNFullConnectRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNFullConnectRelease");
  __BEGIN__;

  CvCNNFullConnectLayer* layer = 0;

  if ( !p_layer )
      CV_ERROR( CV_StsNullPtr, "Null double pointer" );

  layer = *(CvCNNFullConnectLayer**)p_layer;

  if ( !layer )
      return;
  if ( !ICV_IS_CNN_FULLCONNECT_LAYER(layer) )
      CV_ERROR( CV_StsBadArg, "Invalid layer" );

  cvReleaseMat( &layer->exp2ssumWX );
  cvReleaseMat( &layer->weights );
  cvFree( p_layer );

  __END__;
}

static void icvCNNImgCroppingRelease( CvCNNLayer** p_layer ){}

/****************************************************************************************/
static void icvCNNRecurrentRelease( CvCNNLayer** p_layer )
{
  CV_FUNCNAME("icvCNNRecurrentRelease");
  __BEGIN__;

  CvCNNRecurrentLayer* layer = 0;

  if ( !p_layer ) { CV_ERROR( CV_StsNullPtr, "Null double pointer" ); }

  layer = *(CvCNNRecurrentLayer**)p_layer;

  if ( !layer ) { return; }
  if ( !ICV_IS_CNN_RECURRENT_LAYER(layer) ) { 
    CV_ERROR( CV_StsBadArg, "Invalid layer" ); 
  }

  cvReleaseMat( &layer->Wxh );
  cvReleaseMat( &layer->Whh );
  cvReleaseMat( &layer->Why );
  cvFree( p_layer );

  __END__;
}

static void icvCNNInputDataRelease( CvCNNLayer** p_layer ){}

/****************************************************************************************\
*                              Read/Write CNN classifier                                *
\****************************************************************************************/
static int icvIsCNNModel( const void* ptr )
{
  return CV_IS_CNN(ptr);
}

/****************************************************************************************/
static void icvReleaseCNNModel( void** ptr )
{
  CV_FUNCNAME("icvReleaseCNNModel");
  __BEGIN__;

  if ( !ptr ) { CV_ERROR( CV_StsNullPtr, "NULL double pointer" ); }
  CV_ASSERT(CV_IS_CNN(*ptr));

  icvCNNModelRelease( (CvCNNStatModel**)ptr );

  __END__;
}

/****************************************************************************************/
static CvCNNLayer* icvReadCNNLayer( CvFileStorage* fs, CvFileNode* node )
{
  CvCNNLayer* layer = 0;
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

  if ( layer_type == ICV_CNN_CONVOLUTION_LAYER ) {
    const int K = input_height - output_height + 1;
    if ( K <= 0 || K != input_width - output_width + 1 ) {
      CV_ERROR( CV_StsBadArg, "Invalid <K>" );
    }
    CV_CALL(connect_mask = (CvMat*)cvReadByName( fs, node, "connect_mask" ));
    if ( !connect_mask ) {
      CV_ERROR( CV_StsParseError, "Missing <connect mask>" );
    }
    CV_CALL(layer = cvCreateCNNConvolutionLayer( "",
      n_input_planes, input_height, input_width, n_output_planes, K,
      init_learn_rate, learn_type, connect_mask, weights ));
  } else if ( layer_type == ICV_CNN_SUBSAMPLING_LAYER ){
    const int sub_samp_scale = input_height/output_height;
    if ( sub_samp_scale <= 0 || sub_samp_scale != input_width/output_width ) {
      CV_ERROR( CV_StsBadArg, "Invalid <sub_samp_scale>" );
    }
    CV_CALL(layer = cvCreateCNNSubSamplingLayer( "", 
      n_input_planes, input_height, input_width, sub_samp_scale,
      1.f, 1.f, init_learn_rate, learn_type, weights ));
  } else if ( layer_type == ICV_CNN_FULLCONNECT_LAYER ){
    if ( input_height != 1  || input_width != 1 || output_height != 1 || output_width != 1 ) { 
      CV_ERROR( CV_StsBadArg, "" ); 
    }
    CV_CALL(layer = cvCreateCNNFullConnectLayer( "", n_input_planes, n_output_planes,
      1.f, 1.f, init_learn_rate, learn_type, CV_CNN_HYPERBOLIC, weights ));
  } else {
    CV_ERROR( CV_StsBadArg, "Invalid <layer_type>" );
  }

  __END__;

  if ( cvGetErrStatus() < 0 && layer )
      layer->release( &layer );

  cvReleaseMat( &weights );
  cvReleaseMat( &connect_mask );

  return layer;
}

/****************************************************************************************/
static void icvWriteCNNLayer( CvFileStorage* fs, CvCNNLayer* layer )
{
  CV_FUNCNAME ("icvWriteCNNLayer");
  __BEGIN__;

  if ( !ICV_IS_CNN_LAYER(layer) ) { CV_ERROR( CV_StsBadArg, "Invalid layer" ); }

  CV_CALL( cvStartWriteStruct( fs, NULL, CV_NODE_MAP, "opencv-ml-cnn-layer" ));

  CV_CALL(cvWriteInt( fs, "n_input_planes",  layer->n_input_planes ));
  CV_CALL(cvWriteInt( fs, "input_height",    layer->input_height ));
  CV_CALL(cvWriteInt( fs, "input_width",     layer->input_width ));
  CV_CALL(cvWriteInt( fs, "n_output_planes", layer->n_output_planes ));
  CV_CALL(cvWriteInt( fs, "output_height",   layer->output_height ));
  CV_CALL(cvWriteInt( fs, "output_width",    layer->output_width ));
  CV_CALL(cvWriteInt( fs, "learn_rate_decrease_type", layer->learn_rate_decrease_type));
  CV_CALL(cvWriteReal( fs, "init_learn_rate", layer->init_learn_rate ));
  CV_CALL(cvWrite( fs, "weights", layer->weights ));

  if ( ICV_IS_CNN_CONVOLUTION_LAYER( layer )){
    CvCNNConvolutionLayer* l = (CvCNNConvolutionLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_CNN_CONVOLUTION_LAYER ));
    CV_CALL(cvWrite( fs, "connect_mask", l->connect_mask ));
  }else if ( ICV_IS_CNN_SUBSAMPLING_LAYER( layer ) ){
    CvCNNSubSamplingLayer* l = (CvCNNSubSamplingLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_CNN_SUBSAMPLING_LAYER ));
    CV_CALL(cvWriteReal( fs, "a", l->a ));
    CV_CALL(cvWriteReal( fs, "s", l->s ));
  }else if ( ICV_IS_CNN_FULLCONNECT_LAYER( layer ) ){
    CvCNNFullConnectLayer* l = (CvCNNFullConnectLayer*)layer;
    CV_CALL(cvWriteInt( fs, "layer_type", ICV_CNN_FULLCONNECT_LAYER ));
    CV_CALL(cvWriteReal( fs, "a", l->a ));
    CV_CALL(cvWriteReal( fs, "s", l->s ));
  }else {
    CV_ERROR( CV_StsBadArg, "Invalid layer" );
  }

  CV_CALL( cvEndWriteStruct( fs )); //"opencv-ml-cnn-layer"

  __END__;
}

/****************************************************************************************/
static void* icvReadCNNModel( CvFileStorage* fs, CvFileNode* root_node )
{
  CvCNNStatModel* cnn = 0;
  CvCNNLayer* layer = 0;

  CV_FUNCNAME("icvReadCNNModel");
  __BEGIN__;

  CvFileNode* node;
  CvSeq* seq;
  CvSeqReader reader;
  int i;

  CV_CALL(cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
    CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));

  CV_CALL(cnn->etalons = (CvMat*)cvReadByName( fs, root_node, "etalons" ));
  CV_CALL(cnn->cls_labels = (CvMat*)cvReadByName( fs, root_node, "cls_labels" ));

  if ( !cnn->etalons || !cnn->cls_labels ) {
    CV_ERROR( CV_StsParseError, "No <etalons> or <cls_labels> in CNN model" );
  }

  CV_CALL( node = cvGetFileNodeByName( fs, root_node, "network" ));
  seq = node->data.seq;
  if ( !CV_NODE_IS_SEQ(node->tag) ) { CV_ERROR( CV_StsBadArg, "" ); }

  CV_CALL( cvStartReadSeq( seq, &reader, 0 ));
  CV_CALL(layer = icvReadCNNLayer( fs, (CvFileNode*)reader.ptr ));
  CV_CALL(cnn->network = cvCreateCNNetwork( layer ));

  for ( i = 1; i < seq->total; i++ ) {
    CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    CV_CALL(layer = icvReadCNNLayer( fs, (CvFileNode*)reader.ptr ));
    CV_CALL(cnn->network->add_layer( cnn->network, layer ));
  }

  __END__;

  if ( cvGetErrStatus() < 0 ) {
    if ( cnn ) { cnn->release( (CvCNNStatModel**)&cnn ); }
    if ( layer ) { layer->release( &layer ); }
  }
  return (void*)cnn;
}

/****************************************************************************************/
static void icvWriteCNNModel( CvFileStorage* fs, const char* name, 
                              const void* struct_ptr, CvAttrList attr)

{
  CV_FUNCNAME ("icvWriteCNNModel");
  __BEGIN__;

  CvCNNStatModel* cnn = (CvCNNStatModel*)struct_ptr;
  int n_layers, i;
  CvCNNLayer* layer;

  if ( !CV_IS_CNN(cnn) ) { CV_ERROR( CV_StsBadArg, "Invalid pointer" ); }

  n_layers = cnn->network->n_layers;

  CV_CALL( cvStartWriteStruct( fs, name, CV_NODE_MAP, CV_TYPE_NAME_ML_CNN ));

  CV_CALL(cvWrite( fs, "etalons", cnn->etalons ));
  CV_CALL(cvWrite( fs, "cls_labels", cnn->cls_labels ));

  CV_CALL( cvStartWriteStruct( fs, "network", CV_NODE_SEQ ));

  layer = cnn->network->first_layer;
  for ( i = 0; i < n_layers && layer; i++, layer = layer->next_layer ) {
    CV_CALL(icvWriteCNNLayer( fs, layer ));
  }
  if ( i < n_layers || layer ) { CV_ERROR( CV_StsBadArg, "Invalid network" ); }

  CV_CALL( cvEndWriteStruct( fs )); //"network"
  CV_CALL( cvEndWriteStruct( fs )); //"opencv-ml-cnn"

  __END__;
}

static int icvRegisterCNNStatModelType()
{
  CvTypeInfo info;

  info.header_size = sizeof( info );
  info.is_instance = icvIsCNNModel;
  info.release = icvReleaseCNNModel;
  info.read = icvReadCNNModel;
  info.write = icvWriteCNNModel;
  info.clone = NULL;
  info.type_name = CV_TYPE_NAME_ML_CNN;
  cvRegisterType( &info );

  return 1;
} // End of icvRegisterCNNStatModelType

// static int cnn = icvRegisterCNNStatModelType();

#endif

// End of file
