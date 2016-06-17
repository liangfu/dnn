/** -*- mode: c++ -*- 
 *
 * \file   common_layer.h
 * \date   Sat May 14 11:22:07 2016
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
 * \brief  common layer
 */
#ifndef __COMMON_LAYER_H__
#define __COMMON_LAYER_H__

#include "cxcore.h"
#include "list.h"

typedef struct CvCNNLayer CvCNNLayer;

typedef void (CV_CDECL *CvCNNLayerForward)
    ( CvCNNLayer* layer, const CvMat* input, CvMat* output );
typedef void (CV_CDECL *CvCNNLayerBackward)
    ( CvCNNLayer* layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX);
typedef void (CV_CDECL *CvCNNLayerRelease)(CvCNNLayer** layer);

#define CV_CNN_LAYER_FIELDS()                                         \
    /* Indicator of the layer's type */                               \
    int flags;                                                        \
    /* the layer's data type, either CV_32F or CV_64F */              \
    int dtype;                                                        \
    /* Name of the layer, which should be unique within the network*/ \
    char name[20];                                                    \
                                                                      \
    /* Number of input images */                                      \
    int n_input_planes;                                               \
    /* Height of each input image */                                  \
    int input_height;                                                 \
    /* Width of each input image */                                   \
    int input_width;                                                  \
                                                                      \
    /* Number of output images */                                       \
    int n_output_planes;                                                \
    /* Height of each output image */                                   \
    int output_height;                                                  \
    /* Width of each output image */                                    \
    int output_width;                                                   \
    /* current time index */                                            \
    int time_index;                                                     \
    /* sequence length, for attention model: n_glimpses*n_targets */    \
    int seq_length;                                                     \
                                                                        \
    /* activation function type for hidden layer activation, */         \
    /* either sigmoid,tanh,softmax or relu */                           \
    char activation_type[20];                                           \
    /* Learning rate at the first iteration */                          \
    float init_learn_rate;                                              \
    /* Dynamics of learning rate decreasing */                          \
    int decay_type;                                                     \
    /* Dynamics of DELTAw increasing */                                 \
    int delta_w_increase_type;                                          \
    /* samples used in training */                                      \
    int nsamples;                                                       \
    /* max iterations in training */                                    \
    int max_iter;                                                       \
    /* Trainable weights of the layer (including bias) */               \
    /* i-th row is a set of weights of the i-th output plane */         \
    CvMat* weights;                                                     \
    /* Weights matrix from backward pass, for gradient checking */      \
    CvMat * dE_dW;                                                      \
    /* output states, default size: (n_output_planes, batch_size) */    \
    CvMat * Y;                                                          \
                                                                        \
    CvCNNLayerForward  forward;                                         \
    CvCNNLayerBackward backward;                                        \
    CvCNNLayerRelease  release;                                         \
    /* Pointers to the previous and next layers in the network */       \
    CvCNNLayer* prev_layer;                                             \
    CvCNNLayer* next_layer;                                             \
    /* Pointers to the reference layer where weights are taken */       \
    CvCNNLayer* ref_layer;                                              \
    /* Pointers to input/output layer, instead of using prev_layer */   \
    List<CvCNNLayer*> input_layers;                                     \
    List<CvCNNLayer*> output_layers;                                    \
    /* Used in backward pass, in case input_layer is given */           \
    CvMat * dE_dX;                                                      \
    /* Enable memory cache for reducing memory allocation overhead */   \
    CvMat * dY_dX;                                                      \
    int enable_cache;                                                   \
                                                                        \
    int visualize

typedef struct CvCNNLayer
{
    CV_CNN_LAYER_FIELDS();
}CvCNNLayer;

#define ICV_CNN_LAYER                0x55550000
#define ICV_CNN_INPUTDATA_LAYER      0x00001111
#define ICV_CNN_CONVOLUTION_LAYER    0x00002222
#define ICV_CNN_MAXPOOLLING_LAYER    0x00003333
#define ICV_CNN_FULLCONNECT_LAYER    0x00004444
#define ICV_CNN_IMGWARPPING_LAYER    0x00005555
#define ICV_CNN_RECURRENTNN_LAYER    0x00006666
#define ICV_CNN_COMBINATION_LAYER    0x00007777
#define ICV_CNN_SEQSAMPLING_LAYER    0x00008888
#define ICV_CNN_LSTM_LAYER           0x00009999

#define CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY  1
#define CV_CNN_LEARN_RATE_DECREASE_SQRT_INV        2
#define CV_CNN_LEARN_RATE_DECREASE_LOG_INV         3

CV_INLINE
int icvIsCNNLayer( CvCNNLayer * layer ) {
  return ( ((layer) != NULL) &&
           ((((CvCNNLayer*)(layer))->flags & CV_MAGIC_MASK) == ICV_CNN_LAYER ));
}

CV_INLINE
int icvIsCNNConvolutionLayer( CvCNNLayer * layer ) {                              
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*)layer)->flags & ~CV_MAGIC_MASK) == ICV_CNN_CONVOLUTION_LAYER );
}

CV_INLINE
int icvIsCNNMaxPoollingLayer( CvCNNLayer * layer ) {                              
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_MAXPOOLLING_LAYER );
}

CV_INLINE
int icvIsCNNFullConnectLayer( CvCNNLayer * layer ) {                              
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_FULLCONNECT_LAYER );
}

CV_INLINE
int icvIsCNNImgWarppingLayer( CvCNNLayer * layer ) {
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_IMGWARPPING_LAYER );
}

CV_INLINE
int icvIsCNNSeqSamplingLayer( CvCNNLayer * layer ) {
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_SEQSAMPLING_LAYER );
}

CV_INLINE
int icvIsCNNRecurrentNNLayer( CvCNNLayer * layer ) {                              
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_RECURRENTNN_LAYER );
}

CV_INLINE
int icvIsCNNCombinationLayer( CvCNNLayer * layer ) {                              
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_COMBINATION_LAYER );
}

CV_INLINE
int icvIsCNNInputDataLayer( CvCNNLayer * layer ) {                                
  return ( (icvIsCNNLayer( layer )) &&
           (((CvCNNLayer*) (layer))->flags & ~CV_MAGIC_MASK) == ICV_CNN_INPUTDATA_LAYER );
}

typedef struct CvCNNConvolutionLayer
{
  CV_CNN_LAYER_FIELDS();
  // Kernel size (height and width) for convolution.
  int K;
  // for simard method
  CvMat * WX; 
  // (x1+x2+x3+x4), where x1,...x4 are some elements of X
  // - is the vector used in computing of the activation function in backward
  CvMat * sumX;//for simard method
  // connections matrix, (i,j)-th element is 1 iff there is a connection between
  // i-th plane of the current layer and j-th plane of the previous layer;
  // (i,j)-th element is equal to 0 otherwise
  CvMat * connect_mask;
  // value of the learning rate for updating weights at the first iteration
}CvCNNConvolutionLayer;

typedef struct CvCNNMaxPoollingLayer
{
  CV_CNN_LAYER_FIELDS();
  // ratio between the heights (or widths - ratios are supposed to be equal)
  // of the input and output planes
  int sub_samp_scale;
  CvMat * WX;
  // (x1+x2+x3+x4), where x1,...x4 are some elements of X
  // - is the vector used in computing of the activation function in backward
  CvMat * sumX;
  // location where max pooling values are taken from
  CvMat * mask;
}CvCNNMaxPoollingLayer;

// structure of the last layer.
typedef struct CvCNNFullConnectLayer
{
  CV_CNN_LAYER_FIELDS();
  // WX = (W*X) - is the vector used in computing of the 
  // activation function and it's derivative by the formulae
  CvMat * WX;
}CvCNNFullConnectLayer;

typedef struct CvCNNImgWarppingLayer
{
  CV_CNN_LAYER_FIELDS();
  // crop specified time index for next layer
  // int time_index;
}CvCNNImgWarppingLayer;

typedef struct CvCNNRecurrentLayer
{
  CV_CNN_LAYER_FIELDS();
  // reference layer
  // CvCNNLayer * hidden_layer;
  // number of hidden layers within RNN model, default: exp((log(n_inputs)+log(n_outputs))*.5f)
  int n_hiddens;
  // weight matrix for input data, default size: (n_hiddens, n_inputs)
  CvMat * Wxh;
  // weight matrix with bias for hidden data, default size: (n_hiddens, n_hiddens+1)
  CvMat * Whh;
  // weight matrix with bias for generating output data, default size: (n_outputs, n_hiddens+1)
  CvMat * Why;
  // -------------------------------------------------------
  // VARIABLES REQUIRED FOR COMPUTING FORWARD & BACKWARD PASS
  // -------------------------------------------------------
  // gradient in complete sequence, default size: (n_output_planes, batch_size)
  CvMat * dE_dY;                                                     
  // hidden states, default size: (n_hiddens*batch_size, seq_length)
  CvMat * H;
  // input states to hidden states, WX = Wxh*X + Whh*H_prev + bh
  CvMat * WX;
  // hidden states to output states, WH = Why*H_curr + by
  CvMat * WH;
  // accumulated loss for output state loss in complete sequence, loss = \sum_t^T(Y-target)
  double loss;
  // hidden states gradient, size is same as hidden states: (n_hiddens*batch_size, seq_length)
  CvMat * dH;
  // weight updates
  CvMat * dWxh, * dWhh, * dWhy;
}CvCNNRecurrentLayer;

typedef struct CvCNNInputDataLayer
{
  // shape of the data are available in common `layer fields`
  CV_CNN_LAYER_FIELDS();
  // the sequential length of input data
  // int seq_length;
  // original input data, default size: 
  //          (n_input_planes*n_input_width*n_input_height*seq_length, nsamples)
  CvMat * input_data;
  // original response matrix, default size:
  //          (1, nsamples)
  CvMat * response;
}CvCNNInputDataLayer;

typedef struct CvCNNCombinationLayer
{
  CV_CNN_LAYER_FIELDS();
  // CvCNNLayer ** input_layers;
  // int n_input_layers;
}CvCNNCombinationLayer;

/*------------------------ activation functions -----------------------*/
CVAPI(void) cvTanh(CvMat * src, CvMat * dst);
CVAPI(void) cvTanhDer(CvMat * src, CvMat * dst);
CVAPI(void) cvSigmoid(CvMat * src, CvMat * dst);
CVAPI(void) cvSigmoidDer(CvMat * src, CvMat * dst);
CVAPI(void) cvReLU(CvMat * src, CvMat * dst);
CVAPI(void) cvReLUDer(CvMat * src, CvMat * dst);
CVAPI(void) cvSoftmax(CvMat * src, CvMat * dst);
CVAPI(void) cvSoftmaxDer(CvMat * X, CvMat * dE_dY, CvMat * dE_dY_afder);

CVAPI(CvCNNLayer*) cvCreateCNNConvolutionLayer( 
    const int dtype, const char * name, const CvCNNLayer * ref_layer,
    const int visualize, const CvCNNLayer * input_layer, 
    int n_input_planes, int input_height, int input_width, int n_output_planes, int K,
    float init_learn_rate, int update_rule, const char * activation_type,
    CvMat* connect_mask, CvMat* weights );

CVAPI(CvCNNLayer*) cvCreateCNNMaxPoollingLayer( 
    const int dtype, const char * name, const int visualize,
    int n_input_planes, int input_height, int input_width,
    int sub_samp_scale, 
    float init_learn_rate, int update_rule, CvMat* weights );

CVAPI(CvCNNLayer*) cvCreateCNNFullConnectLayer( 
    const int dtype, const char * name, const int visualize,
    const CvCNNLayer * input_layer, int n_inputs, int n_outputs, 
    float init_learn_rate, int update_rule, const char * activation_type,
    CvMat * weights );

CVAPI(CvCNNLayer*) cvCreateCNNImgWarppingLayer( 
    const int dtype, const char * name, const int visualize, 
    const CvCNNLayer * input_layer,
    int n_output_planes, int output_height, int output_width, int seq_length, int time_index,
    float init_learn_rate, int update_rule);

CVAPI(CvCNNLayer*) cvCreateCNNRecurrentLayer( 
    const int dtype, const char * name, const CvCNNLayer * hidden_layer, 
    int n_inputs, int n_outputs, int n_hiddens, int seq_length, int time_index, 
    float init_learn_rate, int update_rule, const char * activation_type, 
    CvMat * Wxh, CvMat * Whh, CvMat * Why );

CVAPI(CvCNNLayer*) cvCreateCNNInputDataLayer( 
    const int dtype, const char * name, 
    int n_inputs, int input_height, int input_width, int seq_length,
    float init_learn_rate, int update_rule);

CVAPI(CvCNNLayer*) cvCreateCNNCombinationLayer( 
    const int dtype, const char * name, const int visualize,
    int n_input_layers, CvCNNLayer ** input_layers, int outputs,
    float init_learn_rate, int update_rule);


#endif // __COMMON_LAYER_H__
