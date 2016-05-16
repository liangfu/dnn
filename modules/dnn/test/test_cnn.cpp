
#include "test_precomp.hpp"

#include <string>
#include <fstream>
#include <iostream>

#include "cvext_c.h"

typedef void (*CvActivationFunc)(CvMat *, CvMat *);
typedef void (*CvActivationDerFunc)(CvMat *, CvMat *, CvMat *);

#define CV_FOREACH_ELEM(mat,ridx,cidx)                                  \
  for(int (ridx)=0;(ridx)<(mat)->rows;(ridx)++)for(int (cidx)=0;(cidx)<(mat)->cols;(cidx)++)

#define CV_NORM_TYPE1 111
#define CV_NORM_TYPE2 222

void cvActivationGradCheck(CvActivationFunc actfunc, CvActivationFunc actfunc_der, const int dtype)
{
  const int nr=100, nc=100;
  const float eps = 1e-4f;
  CvMat * src = cvCreateMat(nr,nc,dtype);
  CvMat * src_more = cvCreateMat(nr,nc,dtype);
  CvMat * src_less = cvCreateMat(nr,nc,dtype);
  CvMat * dst = cvCreateMat(nr,nc,dtype); 
  CvMat * dst_more = cvCreateMat(nr,nc,dtype); 
  CvMat * dst_less = cvCreateMat(nr,nc,dtype); 
  CvMat * diff = cvCreateMat(nr,nc,dtype); 
  CvMat * grad = cvCreateMat(nr,nc,dtype);
  CvMat * src_der = cvCreateMat(nr,nc,dtype);
  CvMat * rel_error = cvCreateMat(nr,nc,dtype);
  CvMat * sum_error = cvCreateMat(nr,nc,dtype);
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
  EXPECT_LT(cvAvg(rel_error).val[0], 1e-5);
  EXPECT_LT(cvNorm(rel_error,0,CV_C), 0.001);
  EXPECT_LT(cvNorm(rel_error,0,CV_L1)/(nr*nc), 0.001);
}

void cvCNNLayerGradCheck(CvCNNLayer * layer, CvMat * X, CvMat * Y, CvMat * target, 
                         CvMat * grad0, CvMat * grad1, int norm_type)
{
  const float eps = 1e-4;
  const int dtype = layer->dtype;
  CvMat * weights = cvCloneMat(layer->weights);
  CvMat * Y_less = cvCreateMat(Y->rows,Y->cols,dtype);
  CvMat * Y_more = cvCreateMat(Y->rows,Y->cols,dtype);
  CvMat * dE_dY = cvCreateMat(Y->cols,Y->rows,dtype);
  CvMat * dE_dX = cvCreateMat(X->cols,X->rows,dtype);
  cvCopy(weights,layer->weights); 
  layer->forward(layer,X,Y); cvTranspose(target,dE_dY); cvAdd(Y,target,target);
  cvScale(dE_dY,dE_dY,-1.f); 
  layer->backward(layer,1,X,dE_dY,dE_dX); cvCopy(layer->dE_dW,grad1);
  for (int ridx=0;ridx<layer->weights->rows;ridx++){
  for (int cidx=0;cidx<layer->weights->cols;cidx++){
    // weights(ridx,cidx) + eps
    cvCopy(weights,layer->weights); 
    cvmSet(layer->weights,ridx,cidx,cvmGet(layer->weights,ridx,cidx)+eps);
    layer->forward(layer,X,Y_more); cvSub(Y_more,target,Y_more); 
    // weights(ridx,cidx) - eps
    cvCopy(weights,layer->weights); 
    cvmSet(layer->weights,ridx,cidx,cvmGet(layer->weights,ridx,cidx)-eps);
    layer->forward(layer,X,Y_less); cvSub(Y_less,target,Y_less);
    double Y_more_loss=0;
    double Y_less_loss=0;
    if (norm_type==CV_NORM_TYPE1){
      CV_FOREACH_ELEM(Y_more,ri,ci){double val=cvmGet(Y_more,ri,ci);Y_more_loss+=val*val;}
      CV_FOREACH_ELEM(Y_less,ri,ci){double val=cvmGet(Y_less,ri,ci);Y_less_loss+=val*val;}
      Y_more_loss*=.5;Y_less_loss*=.5;
    }else if (norm_type==CV_NORM_TYPE2){
      CV_FOREACH_ELEM(Y_more,ri,ci){double val=cvmGet(Y_more,ri,ci);Y_more_loss+=val*val;}
      CV_FOREACH_ELEM(Y_less,ri,ci){double val=cvmGet(Y_less,ri,ci);Y_less_loss+=val*val;}
      Y_more_loss*=2.;Y_less_loss*=2.;
    }else{
      Y_more_loss=cvNorm(Y_more,0,CV_L2);
      Y_less_loss=cvNorm(Y_less,0,CV_L2);
    }
    cvmSet(grad0,ridx,cidx,(Y_more_loss-Y_less_loss)/(2.f*eps));
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

TEST(ML_Tanh, gradcheck){
  cvActivationGradCheck(cvTanh, cvTanhDer, CV_32F);
  cvActivationGradCheck(cvTanh, cvTanhDer, CV_64F);
}
TEST(ML_Sigmoid, gradcheck){
  cvActivationGradCheck(cvSigmoid, cvSigmoidDer, CV_32F);
  cvActivationGradCheck(cvSigmoid, cvSigmoidDer, CV_64F);
}
TEST(ML_ReLU, gradcheck){
  cvActivationGradCheck(cvReLU, cvReLUDer, CV_32F);
  cvActivationGradCheck(cvReLU, cvReLUDer, CV_64F);
}
TEST(ML_Softmax, gradcheck){
  // cvActivationGradCheck(cvSoftmax, cvSoftmaxDer, CV_32F);
  // cvActivationGradCheck(cvSoftmax, cvSoftmaxDer, CV_64F);
}

TEST(ML_ConvolutionLayer, gradcheck){
  const int n_inputs = 2;
  const int n_outputs = 6;
  const int imsize = 20;
  const int ksize = 3;
  const int batch_size = 2;
  const int imsize_out = imsize-ksize+1;
  CvCNNLayer * layer = 
    cvCreateCNNConvolutionLayer(CV_32F,"conv1",0,0,0,n_inputs,imsize,imsize,n_outputs,ksize,.01,1,0,0);
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

void FullConnectLayerTest(int n_inputs, int n_outputs, int batch_size, 
                          int dtype, int norm_type, const char * actype);
TEST(ML_FullConnectLayer, gradcheck){
  int n_inputs = 16;
  int n_outputs = 6;
  int batch_size = 1;
  int dtype = CV_64F;
  FullConnectLayerTest(n_inputs, n_outputs, batch_size, dtype, CV_NORM_TYPE1, "tanh");
  FullConnectLayerTest(n_inputs, n_outputs, batch_size, dtype, CV_NORM_TYPE2, "sigmoid");
  FullConnectLayerTest(n_inputs, n_outputs, batch_size, dtype, CV_NORM_TYPE2, "softmax");
  FullConnectLayerTest(n_inputs, n_outputs, batch_size, dtype, CV_NORM_TYPE1, "relu");
}

void FullConnectLayerTest(int n_inputs, int n_outputs, int batch_size, 
                          int dtype, int norm_type, const char * actype)
{
  CvCNNLayer * layer = 
    cvCreateCNNFullConnectLayer(dtype,"fc1",0,0,n_inputs,n_outputs,.01,1,actype,0);
  ASSERT_TRUE(icvIsCNNFullConnectLayer(layer));
  CvMat * X = cvCreateMat(n_inputs,batch_size,dtype);
  CvMat * Y = cvCreateMat(n_outputs,batch_size,dtype);
  CvMat * target = cvCreateMat(Y->rows,Y->cols,dtype);
  CvMat * grad0 = cvCreateMat(layer->weights->rows,layer->weights->cols,dtype);
  CvMat * grad1 = cvCreateMat(layer->weights->rows,layer->weights->cols,dtype);
  CvMat * norm = cvCreateMat(layer->weights->rows,layer->weights->cols,dtype);
  CvRNG rng = cvRNG(-1);
  cvRandArr(&rng,X,CV_RAND_NORMAL,cvScalar(0),cvScalar(1));
  if (!strcmp(actype,"tanh")){
    cvRandArr(&rng,target,CV_RAND_NORMAL,cvScalar(0),cvScalar(.1));
  }else if (!strcmp(actype,"sigmoid")){
    cvRandArr(&rng,target,CV_RAND_NORMAL,cvScalar(0),cvScalar(.05));
  }else{
    cvRandArr(&rng,target,CV_RAND_NORMAL,cvScalar(0),cvScalar(.05));
  }
  cvCNNLayerGradCheck(layer, X, Y, target, grad0, grad1, norm_type);
  for (int ridx=0;ridx<grad0->rows;ridx++){
  for (int cidx=0;cidx<grad0->cols;cidx++){
    double gval0 = cvmGet(grad0,ridx,cidx);
    double gval1 = cvmGet(grad1,ridx,cidx);
    cvmSet(norm,ridx,cidx,fabs(gval0-gval1));
  }
  }
  fprintf(stderr,"%s: %f\n",actype, cvAvg(norm).val[0]);
  // fprintf(stderr,"\ngrad0:\n");cvPrintf(stderr,"%.2f ",grad0);
  // fprintf(stderr,"\ngrad1:\n");cvPrintf(stderr,"%.2f ",grad1);
  // fprintf(stderr,"\nrel_error:\n");cvPrintf(stderr,"%.2f ",norm);
  fprintf(stderr,"quantile [50%%]:%f\n",cvQuantile(norm,.5));
  fprintf(stderr,"quantile [90%%]:%f\n",cvQuantile(norm,.9));
  fprintf(stderr,"quantile [95%%]:%f\n",cvQuantile(norm,.95));
  fprintf(stderr,"quantile [99%%]:%f\n",cvQuantile(norm,.99));
  EXPECT_LT(cvQuantile(norm,.99),.2f);
  cvReleaseMat(&X);
  cvReleaseMat(&Y);
  cvReleaseMat(&target);
  cvReleaseMat(&grad0);
  cvReleaseMat(&grad1);
  cvReleaseMat(&norm);
}










