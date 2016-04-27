
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
  CvMat * rel_error = cvCreateMat(nr,nc,CV_32F);
  CvMat * sum_error = cvCreateMat(nr,nc,CV_32F);
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
  cvSub(grad,src_der,rel_error); cvAbs(rel_error,rel_error);
  // cvAdd(grad,src_der,sum_error); cvAbs(sum_error,sum_error); cvAddS(sum_error,cvScalar(1e-5),sum_error);
  // cvDiv(rel_error,sum_error,rel_error);
  EXPECT_LT(cvNorm(rel_error,0,CV_C), 0.001);
  EXPECT_LT(cvNorm(rel_error,0,CV_L1)/(nr*nc), 0.001);
}

void cvCNNLayerGradCheck(CvCNNLayer * layer, CvMat * X, CvMat * Y, CvMat * target, 
                         CvMat * grad0, CvMat * grad1, int norm_type)
{
  const float eps = 1e-4;
  CvMat * weights = cvCloneMat(layer->weights);
  CvMat * Y_less = cvCloneMat(Y), * Y_more = cvCloneMat(Y);
  CvMat * dE_dY = cvCreateMat(Y->cols,Y->rows,CV_32F);
  CvMat * dE_dX = cvCreateMat(X->cols,X->rows,CV_32F);
  cvCopy(weights,layer->weights); 
  layer->forward(layer,X,Y); cvTranspose(target,dE_dY); cvAdd(Y,target,target); cvScale(dE_dY,dE_dY,-1.f); 
  layer->backward(layer,1,X,dE_dY,dE_dX); cvCopy(layer->dE_dW,grad1);
  for (int ridx=0;ridx<layer->weights->rows;ridx++){
  for (int cidx=0;cidx<layer->weights->cols;cidx++){
    // weights(ridx,cidx) + eps
    cvCopy(weights,layer->weights); CV_MAT_ELEM(*layer->weights,float,ridx,cidx)+=eps;
    layer->forward(layer,X,Y_more); cvSub(Y_more,target,Y_more); float Y_more_loss=cvNorm(Y_more,0,norm_type);
    // weights(ridx,cidx) - eps
    cvCopy(weights,layer->weights); CV_MAT_ELEM(*layer->weights,float,ridx,cidx)-=eps;
    layer->forward(layer,X,Y_less); cvSub(Y_less,target,Y_less); float Y_less_loss=cvNorm(Y_less,0,norm_type);
    CV_MAT_ELEM(*grad0,float,ridx,cidx) = (Y_more_loss-Y_less_loss)/(2.f*eps);
  }
  }
  cvReleaseMat(&weights);
  cvReleaseMat(&Y_more);
  cvReleaseMat(&Y_less);
  cvReleaseMat(&dE_dY);
  cvReleaseMat(&dE_dX);
}

/////////////////////////////////////////////////////////////////////////////
//////////////////// test registration  /////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

TEST(ML_Tanh, gradcheck){cvActivationGradCheck(cvTanh, cvTanhDer);}
TEST(ML_Sigmoid, gradcheck){cvActivationGradCheck(cvSigmoid, cvSigmoidDer);}
TEST(ML_ReLU, gradcheck){cvActivationGradCheck(cvReLU, cvReLUDer);}
TEST(ML_Softmax, gradcheck){cvActivationGradCheck(cvSoftmax, cvSoftmaxDer);}

TEST(ML_ConvolutionLayer, gradcheck){
  const float eps = 1e-4;
  const int n_inputs = 2;
  const int n_outputs = 6;
  const int imsize = 20;
  const int ksize = 3;
  const int batch_size = 2;
  const int imsize_out = imsize-ksize+1;
  CvCNNLayer * layer = cvCreateCNNConvolutionLayer("conv1",0,n_inputs,imsize,imsize,n_outputs,ksize,.01,1,0,0);
  CvMat * X = cvCreateMat(imsize*imsize*n_inputs,batch_size,CV_32F);
  CvMat * Y = cvCreateMat(imsize_out*imsize_out*n_outputs,batch_size,CV_32F);
  CvMat * target = cvCreateMat(imsize_out*imsize_out*n_outputs,batch_size,CV_32F);
  CvMat * grad0 = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvMat * grad1 = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvMat * norm = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,X,CV_RAND_UNI,cvScalar(-3),cvScalar(3));
  cvRandArr(&rng,target,CV_RAND_NORMAL,cvScalar(0),cvScalar(.1));
  cvCNNLayerGradCheck(layer, X, Y, target, grad0, grad1, CV_L1);
  fprintf(stderr,"\ngrad0:\n");cvPrintf(stderr,"%.2f ",grad0);
  fprintf(stderr,"\ngrad1:\n");cvPrintf(stderr,"%.2f ",grad1);
  for (int ridx=0;ridx<grad0->rows;ridx++){
  for (int cidx=0;cidx<grad0->cols;cidx++){
    float gval0 = CV_MAT_ELEM(*grad0,float,ridx,cidx);
    float gval1 = CV_MAT_ELEM(*grad1,float,ridx,cidx);
    CV_MAT_ELEM(*norm,float,ridx,cidx)=fabs(gval0-gval1)/(fabs(gval0)+fabs(gval1)+1e-5f);
  }
  }
  fprintf(stderr,"\nrel_error:\n");cvPrintf(stderr,"%.2f ",norm);
  fprintf(stderr,"\nquantile [10%%]:%f",cvQuantile(norm,.1));
  fprintf(stderr,"\nquantile [50%%]:%f",cvQuantile(norm,.5));
  fprintf(stderr,"\nquantile [60%%]:%f",cvQuantile(norm,.6));
  fprintf(stderr,"\nquantile [70%%]:%f",cvQuantile(norm,.7));
  fprintf(stderr,"\nquantile [80%%]:%f",cvQuantile(norm,.8));
  fprintf(stderr,"\nquantile [90%%]:%f\n",cvQuantile(norm,.9));
  EXPECT_LT(cvQuantile(norm,.9),.99f);
  cvReleaseMat(&X);
  cvReleaseMat(&Y);
  cvReleaseMat(&target);
  cvReleaseMat(&grad0);
  cvReleaseMat(&grad1);
  cvReleaseMat(&norm);
}

TEST(ML_FullConnectLayer, gradcheck){
  const float eps = 1e-4;
  const int n_inputs = 20;
  const int n_outputs = 10;
  const int batch_size = 2;
  CvCNNLayer * layer = cvCreateCNNFullConnectLayer("fc1",0,0,n_inputs,n_outputs,.01,1,"tanh",0);
  ASSERT_TRUE(ICV_IS_CNN_FULLCONNECT_LAYER(layer));
  CvMat * X = cvCreateMat(n_inputs,batch_size,CV_32F);
  CvMat * Y = cvCreateMat(n_outputs,batch_size,CV_32F);
  CvMat * target = cvCreateMat(Y->rows,Y->cols,CV_32F);
  CvMat * grad0 = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvMat * grad1 = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvMat * norm = cvCreateMat(layer->weights->rows,layer->weights->cols,CV_32F);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,X,CV_RAND_UNI,cvScalar(-2),cvScalar(2));
  cvRandArr(&rng,target,CV_RAND_UNI,cvScalar(-.2),cvScalar(.2));
  cvCNNLayerGradCheck(layer, X, Y, target, grad0, grad1, CV_L2);
  for (int ridx=0;ridx<grad0->rows;ridx++){
  for (int cidx=0;cidx<grad0->cols;cidx++){
    float gval0 = CV_MAT_ELEM(*grad0,float,ridx,cidx);
    float gval1 = CV_MAT_ELEM(*grad1,float,ridx,cidx);
    CV_MAT_ELEM(*norm,float,ridx,cidx)=fabs(gval0-gval1)/(fabs(gval0)+fabs(gval1)+1e-5f);
  }
  }
  fprintf(stderr,"quantile [90%%]:%f\n",cvQuantile(norm,.9));
  fprintf(stderr,"quantile [95%%]:%f\n",cvQuantile(norm,.95));
  EXPECT_LT(cvQuantile(norm,.95),.99f);
  cvReleaseMat(&X);
  cvReleaseMat(&Y);
  cvReleaseMat(&target);
  cvReleaseMat(&grad0);
  cvReleaseMat(&grad1);
  cvReleaseMat(&norm);
}










