
#include "test_precomp.hpp"

#include <string>
#include <fstream>
#include <iostream>

#include "cvext_c.h"

typedef void (*CvActivationFunc)(CvMat *, CvMat *);
// typedef void (CV_CDECL *CvCNNLayerForward)
//     ( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
// typedef void (CV_CDECL *CvCNNLayerBackward)
//     ( CvCNNLayer* layer, int t, const CvMat* X, const CvMat* dE_dY, CvMat* dE_dX);

void cvActivationGradCheck(CvActivationFunc actfunc, CvActivationFunc actfunc_der)
{
  int nr=100, nc=100;
  const float eps = 1e-4f;
  CvMat * src = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_more = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_less = cvCreateMat(nr,nc,CV_32F);
  CvMat * dst = cvCreateMat(nr,nc,CV_32F); 
  CvMat * dst_more = cvCreateMat(nr,nc,CV_32F); 
  CvMat * dst_less = cvCreateMat(nr,nc,CV_32F); 
  CvMat * diff = cvCreateMat(nr,nc,CV_32F); 
  CvMat * grad = cvCreateMat(nr,nc,CV_32F);
  CvMat * src_der = cvCreateMat(nr,nc,CV_32F);
  CvMat * error = cvCreateMat(nr,nc,CV_32F);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,src,CV_RAND_UNI,cvScalar(-5),cvScalar(5));
  cvAddS(src,cvScalar(eps),src_more);
  cvAddS(src,cvScalar(-eps),src_less);
  cvZero(dst); cvZero(dst_more); cvZero(dst_less);
  actfunc(src,dst);
  actfunc(src_more,dst_more);
  actfunc(src_less,dst_less);
  cvSub(dst_more,dst_less,diff);
  cvScale(diff,grad,1./(2.f*eps));
  actfunc_der(src,src_der);
  cvSub(grad,src_der,error);
  EXPECT_LT(cvNorm(error,0,CV_C), 0.001);
  EXPECT_LT(cvNorm(error,0,CV_L1)/(nr*nc), 0.001);
}

void cvCNNLayerGradCheck(CvCNNLayerForward forward, CvCNNLayerBackward backward)
{
}

/////////////////////////////////////////////////////////////////////////////
//////////////////// test registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

TEST(ML_Tanh, gradcheck){cvActivationGradCheck(cvTanh, cvTanhDer);}
TEST(ML_Sigmoid, gradcheck){cvActivationGradCheck(cvSigmoid, cvSigmoidDer);}
TEST(ML_ReLU, gradcheck){cvActivationGradCheck(cvReLU, cvReLUDer);}
TEST(ML_Softmax, gradcheck){cvActivationGradCheck(cvSoftmax, cvSoftmaxDer);}

// CVAPI(CvCNNLayer*) cvCreateCNNConvolutionLayer( const char * name, const int visualize,
//     int n_input_planes, int input_height, int input_width,
//     int n_output_planes, int K,
//     float init_learn_rate, int learn_rate_decrease_type,
//     CvMat* connect_mask, CvMat* weights );
TEST(ML_ConvolutionLayer, gradcheck){
  const float eps = 1e-4;
  const int n_inputs = 6;
  const int n_outputs = 16;
  const int imsize = 28;
  const int ksize = 5;
  const int batch_size = 2;
  const int imsize_out = imsize-ksize+1;
  CvCNNLayer * layer = cvCreateCNNConvolutionLayer("conv1",0,6,imsize,imsize,16,ksize,.01,1,0,0);
  CvMat * X = cvCreateMat(imsize*imsize*n_inputs,batch_size,CV_32F);
  CvMat * Y = cvCreateMat(imsize_out*imsize_out*n_outputs,batch_size,CV_32F);
  CvMat * X_less = cvCloneMat(X), * X_more = cvCloneMat(X);
  CvMat * Y_less = cvCloneMat(Y), * Y_more = cvCloneMat(Y);
  CvMat * grad0 = cvCloneMat(Y), * grad1 = cvCloneMat(Y), * error = cvCloneMat(Y);
  CvMat * dE_dY = cvCreateMat(Y->cols,Y->rows,CV_32F);
  CvMat * dE_dX = cvCreateMat(X->cols,X->rows,CV_32F);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,X,CV_RAND_UNI,cvScalar(-5),cvScalar(5));
  cvAddS(X,cvScalar(eps),X_more);
  cvAddS(X,cvScalar(-eps),X_less);
  cvZero(Y); cvZero(Y_more); cvZero(Y_less);
  layer->forward(layer,X,Y); cvTranspose(Y,dE_dY);
  layer->forward(layer,X_more,Y_more);
  layer->forward(layer,X_less,Y_less);
  cvSub(Y_more,Y_less,grad0);
  cvScale(grad0,grad0,1./(2.f*eps));
  layer->backward(layer,1,X,dE_dY,dE_dX); cvTranspose(dE_dX,grad1);
  cvSub(grad0,grad1,error);
  EXPECT_LT(cvNorm(error,0,CV_C), 0.001);
  EXPECT_LT(cvNorm(error,0,CV_L1)/(error->rows*error->cols), 0.001);
}










