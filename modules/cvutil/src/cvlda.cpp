/**
 * @file   cvlda.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Apr  2 08:50:17 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvlda.h"

void icvMean(CvMat * src, CvMat * dst)
{
  assert(CV_MAT_TYPE(src->type)==CV_32F);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  float * srcptr = src->data.fl;
  float * dstptr = dst->data.fl;
  int i,j,nr=src->rows,nc=src->cols;
  int step=src->step/sizeof(float);
  assert(step==dst->step/sizeof(float));
  assert((dst->rows==1)&&(dst->rows==nr));
  cvZero(dst);
  for (i=0;i<nr;i++){
  for (j=0;j<nc;j++){
    dstptr[j]+=srcptr[j];
  }
  srcptr+=step;
  }
  cvScale(dst, dst, 1./float(nr));
}

int CvLDA::train(CvMat * train_data, CvMat * response)
{
  int i,j,k,K=0,classes[100]={0,},N=response->rows,D=train_data->cols;
  assert(response->rows==train_data->rows);
  assert(response->cols==1);
  assert(CV_MAT_TYPE(train_data->type)==CV_32F);
  assert(CV_MAT_TYPE(response->type)==CV_32S);
  //---------------------------------
  // get unique categories of training data
  {
    int * resptr = response->data.i;
    classes[K++]=resptr[0];
    for (i=0;i<N;i++){
      for (j=0;j<K;j++){
        if (classes[j]==resptr[i]){break;}
      }
      if (j==K){classes[K++]=resptr[i];}
    }
    //fprintf(stderr, "%d\n", K);
  }

  // K components to classify
  typedef struct {CvMat * mu;CvMat * sigma;int N;} IcvComponent;
  IcvComponent * comps = new IcvComponent[K];
  
  for (k=0;k<K;k++)
  {
    comps[k].mu=cvCreateMat(1,D,CV_32F); 
    comps[k].sigma=cvCreateMat(D,D,CV_32F);
    comps[k].N=0;
  }

  //---------------------------------
  // mean and covariance of each class
  for (k=0;k<K;k++)
  {
    //---------------------------------
    // compute mean
    int * resptr=response->data.i;
    float * muptr=comps[k].mu->data.fl;
    float * datptr=train_data->data.fl;
    comps[k].N=0;
    int step=train_data->step/sizeof(float);
    cvZero(comps[k].mu);
    for (i=0;i<N;i++){
      if (resptr[i]==classes[k]){
        comps[k].N++;
        for (j=0;j<D;j++){
          muptr[j]+=datptr[j];
        }
      }
      datptr+=step;
    }
    cvScale(comps[k].mu,comps[k].mu,1./float(comps[k].N));
    // fprintf(stderr, "%d\n", comps[k].N);
    // cvPrintf(stderr, "%.2f,", comps[k].mu);
    //---------------------------------
    // compute covariance
    CvMat * meansub = cvCreateMat(comps[k].N, D, CV_32F);
    float * meansubptr = meansub->data.fl;
    muptr=comps[k].mu->data.fl;
    datptr=train_data->data.fl;
    int meansubstep=meansub->step/sizeof(float);
    for (i=0;i<N;i++){
      if (resptr[i]==classes[k]){
        for (j=0;j<D;j++){
          meansubptr[j]=datptr[j]-muptr[j];
        }
        meansubptr+=meansubstep;
      }
      datptr+=step;
    }
    cvGEMM(meansub,meansub,1,0,1,comps[k].sigma,CV_GEMM_A_T);
    // cvScale(comps[k].sigma,comps[k].sigma,1./float(comps[k].N));
    // fprintf(stderr, "covar:\n");
    // cvPrintf(stderr, "%.2f,", comps[k].sigma);
    cvReleaseMat(&meansub);
  }
        
  //---------------------------------
  // within-class scatter
  CvMat * Sw = cvCreateMat(D,D,CV_32F);
  CvMat * invSw = cvCreateMat(D,D,CV_32F);
  cvZero(Sw);
  for (k=0;k<K;k++) { cvAdd(Sw,comps[k].sigma,Sw); }
  cvInvert(Sw,invSw,CV_SVD);
  // cvPrintf(stderr, "%.2f,",invSw, cvRect(0,0,6,6));

  //---------------------------------
  // compute matrix W for classification
  if (!W) { W = cvCreateMat(K, D+1, CV_32F); }
  else { cvReleaseMat(&W); W = cvCreateMat(K, D+1, CV_32F); }
  
  CvMat * tmp = cvCreateMat(1, D, CV_32F);
  float * wptr = W->data.fl;
  float * invSwptr = invSw->data.fl;
  int wstep=W->step/sizeof(float);
  for (k=0;k<K;k++) {
    cvMatMul(comps[k].mu,invSw,tmp);
    double tval=0;
    for (i=0;i<D;i++) {
      tval += (comps[k].mu->data.fl[i]*tmp->data.fl[i]); }
    wptr[0]=-0.5*tval+log(float(comps[k].N)/float(N));
    memcpy(wptr+1,tmp->data.fl,sizeof(float)*D);
    wptr+=wstep;
  }
  cvPrintf(stderr, "%.2f,", W);
  cvReleaseMat(&tmp);

  //---------------------------------
  // compute probability - perform validation
  // CvMat * L = cvCreateMat(N,K,CV_32F);
  // CvMat * expL = cvCreateMat(N,K,CV_32F);
  // CvMat * newX = cvCreateMat(N,D+1,CV_32F);
  // cvSet(newX,cvScalar(1));
  // float * nXptr=newX->data.fl;
  // float * Xptr =train_data->data.fl;
  // int nXstep = newX->step/sizeof(float);
  // int  Xstep = train_data->step/sizeof(float);
  // for (i=0;i<N;i++){
  //   memcpy(nXptr+1,Xptr,sizeof(float)*D);
  //   nXptr+=nXstep;Xptr+=Xstep;
  // }
  // cvGEMM(newX,W,1,0,1,L,CV_GEMM_B_T);
  // cvPrintf(stderr, "%.2f,", L);
  // cvExp(L, expL);
  // {
  //   float * expLptr = expL->data.fl;
  //   int expLstep = expL->step/sizeof(float);
  //   double sum=0;
  //   for (i=0;i<N;i++){
  //     sum=0;
  //     for (j=0;j<K;j++){
  //       sum+=expLptr[j];
  //     }
  //     for (j=0;j<K;j++){
  //       expLptr[j]/=sum;
  //     }
  //     expLptr+=expLstep;
  //   }
  // }
  // cvPrintf(stderr, "%.2f,", expL);
  // cvReleaseMat(&L);
  // cvReleaseMat(&expL);
  // cvReleaseMat(&newX);
  
  for (k=0;k<K;k++){
    cvReleaseMat(&comps[k].mu);
    cvReleaseMat(&comps[k].sigma);
  }
  delete [] comps;
  cvReleaseMat(&Sw);
  cvReleaseMat(&invSw);
  // cvReleaseMat(&W);W=NULL;
  return 1;
}

int CvLDA::predict(CvMat * sample, CvMat * expL)
{
  float retval=-1;
  assert(W);

  int N=sample->rows,K=2,D=sample->cols,i,j;
  assert(W->rows==K);
  assert(W->cols==(D+1));
  assert(CV_MAT_TYPE(W->type)==CV_32F);
  // CvMat W=cvMat(K,D+1,CV_32F,W_data);

  CvMat * L = cvCreateMat(N,K,CV_32F);
  //CvMat * expL = cvCreateMat(N,K,CV_32F);
  assert((expL->rows==N)&&(expL->cols==K));
  assert(CV_MAT_TYPE(expL->type)==CV_32F);
  CvMat * newX = cvCreateMat(N,D+1,CV_32F);
  cvSet(newX,cvScalar(1));

  // copy sample data to newX, starting from 2nd column
  {
    float * nXptr=newX->data.fl;
    float * Xptr =sample->data.fl;
    int nXstep = newX->step/sizeof(float);
    int  Xstep = sample->step/sizeof(float);
    for (i=0;i<N;i++){
      memcpy(nXptr+1,Xptr,sizeof(float)*D);
      nXptr+=nXstep;Xptr+=Xstep;
    }
  }

  cvGEMM(newX,W,1,0,1,L,CV_GEMM_B_T);
  // cvPrintf(stderr, "%.2f,", L);
  cvExp(L, expL);

  {
    float * expLptr = expL->data.fl;
    int expLstep = expL->step/sizeof(float);
    double sum=0;
    for (i=0;i<N;i++){
      sum=0;
      for (j=0;j<K;j++){
        sum+=expLptr[j];
      }
      for (j=0;j<K;j++){
        expLptr[j]/=sum;
      }
      expLptr+=expLstep;
    }
  }
  // cvPrintf(stderr, "%.2f,", expL);

  // release allocated memory
  cvReleaseMat(&L);
  // cvReleaseMat(&expL);
  cvReleaseMat(&newX);

  return retval;
}

/** 
 *   compute probability
 * 
 *   L = X*W';
 *   P = zeros(size(L,1),2);
 *   for i=1:K
 *     tmp = exp(-0.5*sum((L-repmat(Prior(i,:),[sumN,1])).^2,2));
 *     P(:,i) = tmp(:,1);
 *   end
 *   P = P./repmat(sum(P,2),[1,size(P,2)]);
 * 
 * @param sample 
 * @param result 
 * 
 * @return status code
 */
int CvLDA::predict_withprior(CvMat * sample, CvMat * result)
{
  int retval=-1;
  assert(sample);
  assert(result);
  assert(W);
  assert(PRIOR);

  CvMat * L = cvCreateMat(sample->rows,W->rows,CV_32F);
  cvGEMM(sample,W,1,0,1,L,CV_GEMM_B_T);
  // cvPrintf(stderr, "%.2f,", sample,cvRect(0,0,5,1));
  // cvPrintf(stderr, "%.2f,", L);
  cvSet(result, cvScalar(0));

  int i,j;
  {
    double sumval[2]={0,0};
    float * Lptr = L->data.fl; int Lstep=L->step/sizeof(float);
    float * PRIORptr = PRIOR->data.fl;
    int PRIORstep = PRIOR->step/sizeof(float);
    assert(L->rows==1);
    // for (i=0;i<L->cols;i++){
    for (i=0;i<1;i++){
      sumval[0]+=pow(Lptr[i]-PRIORptr[i],2);
    }
    PRIORptr+=PRIORstep;
    // for (i=0;i<L->cols;i++){
    for (i=0;i<1;i++){
      sumval[1]+=pow(Lptr[i]-PRIORptr[i],2);
    }
    sumval[0]=exp(-0.5*sumval[0]);
    sumval[1]=exp(-0.5*sumval[1]);
    double invsumsumval = 1./(sumval[0]+sumval[1]);
    result->data.fl[0]=sumval[0]*invsumsumval;
    result->data.fl[1]=sumval[1]*invsumsumval;
    retval = 1;
  }
  
  return retval;
}

