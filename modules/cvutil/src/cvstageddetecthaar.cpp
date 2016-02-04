/**
 * @file   cvfacedetector.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jul  8 15:55:57 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvstageddetecthaar.h"

#define CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL 1

CvMat * icvCreateHaarFeatureSetEx(int tsize);

void icvWeightedThresholdClassify_v0(CvMat ** evalarr,
                                     double & thres, double & p)
{
  assert(CV_MAT_TYPE(evalarr[0]->type)==CV_32F);
  assert(CV_MAT_TYPE(evalarr[1]->type)==CV_32F);
  int i;
  double mu[2],minval,maxval;
  for (i=0;i<2;i++){mu[i]=cvAvg(evalarr[i]).val[0];}
  cvMinMaxLoc(evalarr[1],&minval,&maxval);
  if (mu[1]<mu[0]) {thres=maxval;p=1;} else {thres=minval;p=-1;}
}

void icvWeightedThresholdClassify_v1(CvMat ** evalarr,
                                     double & thres, double & p)
{
  assert(CV_MAT_TYPE(evalarr[0]->type)==CV_32F);
  assert(CV_MAT_TYPE(evalarr[1]->type)==CV_32F);
  int i,j,cols[2]={evalarr[0]->cols,evalarr[1]->cols};
  double ratio,sum,var[2],mu[2],minval=0xffffff,maxval=-0xfffff;
  float * evalptr[2];
  evalptr[0] = evalarr[0]->data.fl;
  evalptr[1] = evalarr[1]->data.fl;
  for (i=0;i<2;i++){
    sum=0;
    for (j=0;j<cols[i];j++){sum+=evalptr[i][j];}
    mu[i]=sum/double(cols[i]);
    sum=0;
    for (j=0;j<cols[i];j++){sum+=pow(evalptr[i][j]-mu[i],2.);}
    var[i]=sqrt(sum)/double(cols[i]);
  }
  for (j=0;j<cols[1];j++){
    minval=MIN(evalptr[1][j],minval);
    maxval=MAX(evalptr[1][j],maxval);
  }
  if (mu[1]<mu[0]) {
    ratio=var[1]/(var[0]+var[1]);//ratio=pow(ratio,1.1);
    thres=MIN(maxval,mu[1]+(mu[0]-mu[1])*ratio);p=1;
  } else {
    ratio=var[0]/(var[0]+var[1]);//ratio=pow(ratio,1.1);
    thres=MAX(minval,mu[1]-(mu[1]-mu[0])*ratio);p=-1;
  }
}

void icvWeightedThresholdClassify_v2(CvMat ** evalarr,
                                     double & thres, double & polar)
{
  assert(CV_MAT_TYPE(evalarr[0]->type)==CV_32F);
  assert(CV_MAT_TYPE(evalarr[1]->type)==CV_32F);
  double mu[2],minval,maxval;
  int i,j;float fi=1,di=1,ratio=.98;int fcc=0,dcc=0;
  float * evalresptr;
  for (i=0;i<2;i++){mu[i]=cvAvg(evalarr[i]).val[0];}
  cvMinMaxLoc(evalarr[1],&minval,&maxval);
  if (mu[1]<mu[0]) {
    thres=maxval;polar=1;
    while (di>ratio){
fcc=0;dcc=0;
thres-=1;
for (i=0;i<2;i++){
evalresptr=evalarr[i]->data.fl;
for (j=0;j<evalarr[i]->cols;j++){
if ((evalresptr[j]*polar<thres*polar)&&(i==0)) {fcc++;}
if ((evalresptr[j]*polar<thres*polar)&&(i==1)) {dcc++;}
}
}
fi=double(fcc)/double(evalarr[0]->cols);
di=double(dcc)/double(evalarr[1]->cols);
    }
  } else {
    thres=minval;polar=-1;
    while (di>ratio){
fcc=0;dcc=0;
thres+=1;
for (i=0;i<2;i++){
evalresptr=evalarr[i]->data.fl;
for (j=0;j<evalarr[i]->cols;j++){
if ((evalresptr[j]*polar<thres*polar)&&(i==0)) {fcc++;}
if ((evalresptr[j]*polar<thres*polar)&&(i==1)) {dcc++;}
}
}
fi=double(fcc)/double(evalarr[0]->cols);
di=double(dcc)/double(evalarr[1]->cols);
    }
  }
}

CV_INLINE
double icvEval(CvMat * imgIntegral, float * ftsptr)
{
  int * integral_data=imgIntegral->data.i;
  int i,step = imgIntegral->step/sizeof(int);
  int xx,yy,ww,hh,wt[3],p0[3],p1[3],p2[3],p3[3];
  double fval;
  for (i=0;i<3;i++){
    xx=cvRound((ftsptr+5*i)[0]);
    yy=cvRound((ftsptr+5*i)[1]);
    ww=cvRound((ftsptr+5*i)[2]);
    hh=cvRound((ftsptr+5*i)[3]);
    wt[i]=cvRound((ftsptr+5*i)[4]);
    if (i==2){if(wt[2]==0){break;}}
    p0[i]=xx+yy*step;
    p1[i]=xx+ww+yy*step;
    p2[i]=xx+(yy+hh)*step;
    p3[i]=xx+ww+(yy+hh)*step;
  }
  if (wt[2]==0){
    fval =
        (integral_data[p0[0]]-integral_data[p1[0]]-
         integral_data[p2[0]]+integral_data[p3[0]])*wt[0]+
        (integral_data[p0[1]]-integral_data[p1[1]]-
         integral_data[p2[1]]+integral_data[p3[1]])*wt[1];
  }else{
    fval =
        (integral_data[p0[0]]-integral_data[p1[0]]-
         integral_data[p2[0]]+integral_data[p3[0]])*wt[0]+
        (integral_data[p0[1]]-integral_data[p1[1]]-
         integral_data[p2[1]]+integral_data[p3[1]])*wt[1]+
        (integral_data[p0[2]]-integral_data[p1[2]]-
         integral_data[p2[2]]+integral_data[p3[2]])*wt[2];
  }
  return fval;
}

CV_INLINE
int icvIsTarget(CvMat * imgIntegral, CvRect roi, float * h, const int tsize)
{
  double scale = float(roi.width)/float(tsize);
  int i,step = imgIntegral->step/sizeof(int);
  int xx,yy,ww,hh,wt[3],p0[3],p1[3],p2[3],p3[3];
  double threshold=h[0];
  double polarity =h[1];
  for (i=0;i<3;i++){
    xx=cvRound((h+3+5*i)[0]*scale);
    yy=cvRound((h+3+5*i)[1]*scale);
    ww=cvRound((h+3+5*i)[2]*scale);
    hh=cvRound((h+3+5*i)[3]*scale);
    wt[i]=cvRound((h+3+5*i)[4]);
    if (i==2){if(wt[2]==0){break;}}
    p0[i]=xx+roi.x+(yy+roi.y)*step;
    p1[i]=xx+ww+roi.x+(yy+roi.y)*step;
    p2[i]=xx+roi.x+(yy+hh+roi.y)*step;
    p3[i]=xx+ww+roi.x+(yy+hh+roi.y)*step;
  }
  int * integral_data=imgIntegral->data.i;
  double fval;
  if (wt[2]==0){
    fval =
        (integral_data[p0[0]]-integral_data[p1[0]]-
         integral_data[p2[0]]+integral_data[p3[0]])*wt[0]+
        (integral_data[p0[1]]-integral_data[p1[1]]-
         integral_data[p2[1]]+integral_data[p3[1]])*wt[1];
  }else{
    fval =
        (integral_data[p0[0]]-integral_data[p1[0]]-
         integral_data[p2[0]]+integral_data[p3[0]])*wt[0]+
        (integral_data[p0[1]]-integral_data[p1[1]]-
         integral_data[p2[1]]+integral_data[p3[1]])*wt[1]+
        (integral_data[p0[2]]-integral_data[p1[2]]-
         integral_data[p2[2]]+integral_data[p3[2]])*wt[2];
  }

  return ((fval*polarity)<(threshold*pow(scale,2)*polarity))?1:0;
}

int CvStagedDetectorHaar::detect(CvMat * img, CvRect roi[])
{
  // assert(m_classifier);
  // assert(m_features);
  int nr=img->rows,nc=img->cols;
  CvMat * imgIntegral=cvCreateMat(nr+1,nc+1,CV_32S);
  cvIntegral(img,imgIntegral);
  int i,j,k,cc=0,maxcc=8000;

#if 0
  CvMat * haarclassifier = get_haarclassifier_face20_v0();
  int winsz,tsize=20;
  int num_classifiers=haarclassifier->rows;
  int nstages=14,stageiter;
  int stages_data[]={0,1,8,21,33,42,62,97,148,267,308,339,361,449,986,
                     num_classifiers};
  double thres00=.95,thres11=.99;
#else
  CvMat * haarclassifier = get_haarclassifier_face20_v1();
  int winsz,tsize=20;
  int num_classifiers=haarclassifier->rows;
  int nstages=10,stageiter;
  int stages_data[]={0,1,2,10,29,73,103,160,196,201,316,num_classifiers};
  double thres00=.85,thres11=.96;
#endif
  CvRect currroi;
  float h[18];double sum0,sum1,sum00,sum11;
  for (winsz=tsize;winsz<float(MIN(nr,nc))*.6;
       winsz=cvRound(float(winsz)*1.25)){
  for (i=0;i<nr-winsz;i=cvRound(float(i)+float(winsz)/10.)){
  for (j=0;j<nc-winsz;j=cvRound(float(j)+float(winsz)/10.)){

  // staged rejection
  currroi=cvRect(j,i,winsz,winsz);
  sum00=0;sum11=0;
  for (stageiter=0;stageiter<nstages;stageiter++){  
  sum0=0;sum1=0;
  for (k=stages_data[stageiter];
       k<MIN(num_classifiers,stages_data[stageiter+1]);k++)
  {
    // extract classifier
    memcpy(h,haarclassifier->data.ptr+sizeof(h)*k,sizeof(h));
    // examin the target
    sum0+=h[2]*float(icvIsTarget(imgIntegral,currroi,h,tsize));
    sum1+=h[2];
  } // k loop
  sum00+=sum0;sum11+=sum1;
  // if ((sum0<.99*sum1)||(k>=num_classifiers)){break;}
  if ((sum0<thres00*sum1)||(k>=num_classifiers)){break;}
  } // stageiter loop
  if ((sum00>=thres11*sum11)&&(k==num_classifiers)){roi[cc++]=currroi;}
  
  if(cc>=maxcc){break;}
  }
  if(cc>=maxcc){break;}
  }
  }
  
  cvReleaseMat(&imgIntegral);
  
  return cc;
}

int CvStagedDetectorHaar::
train(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
      int maxiter, int startiter)
{
  typedef void (*CvClassifierFuncType)(CvMat **, double &, double &);
  static CvClassifierFuncType icvWeightedThresholdClassify[3]={
    icvWeightedThresholdClassify_v0,
    icvWeightedThresholdClassify_v1,
    icvWeightedThresholdClassify_v2
  };
  
  int i,j,k,iter;
  int tsize = posimgs[0]->cols+1;
  if (tsize!=(posimgs[0]->rows+1)){return -1;}
  CvMat ** imgIntegral[2];
  CvMat * evalres[2];for(i=0;i<2;i++){evalres[i]=0;}
  int m=nneg,l=npos;
  // int m=2000,l=1000;
  int count[2]={m,l};
  if (!features) {
    features = icvCreateHaarFeatureSetEx(tsize);
    fprintf(stderr,"numfeatures: %d\n",features->rows);
  }
// cvPrintf(stderr,"%.0f,",features,cvRect(0,0,features->cols,5));
  int nfeatures = features->rows,nsamples=m+l;
  if (!weights) {
    // weights = cvCreateMat((startiter==0)?8000:maxiter+1,nsamples,CV_64F);
    weights = cvCreateMat(1,nsamples,CV_64F);
  }
  double wtsum,invwtsum; double * wtptr; double epsval;
  for (i=0;i<2;i++){ imgIntegral[i]=new CvMat *[count[i]]; }
#if CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
  for (i=0;i<2;i++){
    if (!evalres_precomp[i]){
      evalres_precomp[i]=cvCreateMat(nfeatures,count[i],CV_32F);
      fprintf(stderr,"INFO: precompute eval[%d] allocated!\n",i);
    }
  }
#endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
  for (i=0;i<2;i++){ evalres[i]=cvCreateMat(1,count[i],CV_32F); }

  // compute integral images
  for (i=0;i<2;i++){
  for (j=0;j<count[i];j++){
    imgIntegral[i][j]=cvCreateMat(tsize,tsize,CV_32S);
    cvIntegral(i==0?negimgs[j]:posimgs[j],imgIntegral[i][j]);
  }
  }

  // initialize weights
  if (!weights_initialized){
  for (i=0;i<  m;i++){weights->data.db[i]=.5/double(m);} // negative 
  for (i=m;i<l+m;i++){weights->data.db[i]=.5/double(l);} // positive
  weights_initialized=1;
  }
  
#if CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
  // precompute evaluation results
  if (!evalres_precomputed){
  for (i=0;i<nfeatures;i++){
    float * ftsptr = features->data.fl+i*features->cols;
    // extract feature values from integral images
    for (j=0;j<2;j++){
      for (k=0;k<count[j];k++){
        CV_MAT_ELEM(*evalres_precomp[j],float,i,k)=
            icvEval(imgIntegral[j][k],ftsptr+3);
      }
    }if((i%10000)==1){fprintf(stderr,"-");}
  }fprintf(stderr,"\n");
  fprintf(stderr,"INFO: precompute eval complete!\n");
  evalres_precomputed=1;
  }
#endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL

  // find a single feature in each loop
  // for (iter=startiter;iter<maxiter;iter++)
  iter=startiter;
  {
    CvMat * epsilon = cvCreateMat(1,nfeatures,CV_64F);
    cvSet(epsilon,cvScalar(-1));
    
    // normalize weights
    {
    wtsum=0;
    wtptr = weights->data.db;//+nsamples*iter;
    for (i=0;i<nsamples;i++){wtsum+=wtptr[i];}
    invwtsum=1./wtsum;
    for (i=0;i<nsamples;i++){wtptr[i]=wtptr[i]*invwtsum;}
    }

    // for (i=iter;i<nfeatures;i++)
    for (i=0;i<nfeatures;i++)
    {
      float * ftsptr = features->data.fl+i*features->cols;
      // extract feature values from integral images
      for (j=0;j<2;j++){
#if CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
        memcpy(evalres[j]->data.ptr,
               evalres_precomp[j]->data.ptr+i*evalres_precomp[j]->step,
               sizeof(float)*count[j]);
		assert(evalres_precomp[j]->step==sizeof(float)*count[j]);
#else
      for (k=0;k<count[j];k++){
        evalres[j]->data.fl[k]=icvEval(imgIntegral[j][k],ftsptr+3);
      }
#endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
      }
      // approximate classifier parameters
      double threshold,polarity;
      icvWeightedThresholdClassify[1](evalres,threshold,polarity);
      ftsptr[0]=threshold;ftsptr[1]=polarity;

      // compute classification error
      epsval=0;
      for (j=0;j<2;j++){
      float * evalresptr = evalres[j]->data.fl;
      double * wt0ptr = wtptr+count[0]*j;
      for (k=0;k<count[j];k++){
        epsval+=wt0ptr[k]*
		  fabs(double((((evalresptr[k]*polarity)<(threshold*polarity))?1:0)-j));
		// assert(((((evalresptr[k]*polarity)<(threshold*polarity))?1:0)-j)==0||
		// 	      ((((evalresptr[k]*polarity)<(threshold*polarity))?1:0)-j)==1);
      }
      }
      epsilon->data.db[i]=epsval;
    } // end of i loop
    
    // min error feature classifier
    // double minval=epsilon->data.db[iter]; int minloc;
    // for (i=iter;i<nfeatures;i++){
    //   double epsval = epsilon->data.db[i];
    //   if (epsval<minval){minval=epsval;minloc=i;}
    // }
    double minval=epsilon->data.db[0]; int minloc;
    for (i=0;i<nfeatures;i++){
      double epsval = epsilon->data.db[i];
      if (epsval<minval){minval=epsval;minloc=i;}
    }
	selected.push_back(minloc);
//     swap(features->data.ptr+features->step*iter,
//          features->data.ptr+features->step*minloc,
//          features->step);
// #if CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
//     for (j=0;j<2;j++){
//     swap(evalres_precomp[j]->data.ptr+evalres_precomp[j]->step*iter,
//          evalres_precomp[j]->data.ptr+evalres_precomp[j]->step*minloc,
//          evalres_precomp[j]->step);
//     }
// #endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
    
    // update the weights
    // {
    // double * wt0ptr = weights->data.db+nsamples*iter;
    // double * wt1ptr = weights->data.db+nsamples*(iter+1);
    // float thres = (features->data.fl+iter*features->cols)[0];
    // float polar = (features->data.fl+iter*features->cols)[1];
    // double beta = minval/(1.-minval),ei;
    // (features->data.fl+iter*features->cols)[2]=log(1./(beta+1e-6)); // alpha
    // k=0;
    // for (i=0;i<2;i++){
    // float * evalresptr = evalres[i]->data.fl;
    // for (j=0;j<count[i];j++,k++){
    //   ei = ((evalresptr[j]*polar)<(thres*polar))?1:0;
    //   wt1ptr[k]=wt0ptr[k]*pow(beta,ei);
    // }
    // }
	// assert(k==nsamples);
    // }
	
    {
	// CvMat * tmp = cvCreateMat(weights->rows,weights->cols,CV_64F);
	CvMat * tmp = cvCreateMat(1,nsamples,CV_64F);
	cvCopy(weights,tmp);  
    double * wt0ptr = tmp->data.db;
    double * wt1ptr = weights->data.db;
    float thres = (features->data.fl+minloc*features->cols)[0];
    float polar = (features->data.fl+minloc*features->cols)[1];
    double beta = minval/(1.-minval),ei;
	float * evalresptr;
    //(features->data.fl+minloc*features->cols)[2]=log(1./(beta+1e-6)); // alpha
    (features->data.fl+minloc*features->cols)[2]=log(1./beta); // alpha
    k=0;
    for (i=0;i<2;i++){
    evalresptr = evalres_precomp[i]->data.fl+minloc*evalres_precomp[i]->cols;
	assert(evalres_precomp[i]->cols*sizeof(float)==evalres_precomp[i]->step);
    for (j=0;j<count[i];j++,k++){
      //ei = ((evalresptr[j]*polar)<(thres*polar))?1:0;
      ei = ((evalresptr[j]*polar)<(thres*polar))?0:1;
      wt1ptr[k]=wt0ptr[k]*pow(beta,ei);
    }
    }
	assert(k==nsamples);
	cvReleaseMat(&tmp);
    }
	
	if (selected.size()){
	  cvPrintf(stderr,"%f,",epsilon,cvRect(minloc-2,0,5,1));
	  cvPrintf(stderr,"%f,",weights,cvRect(100,0,5,1));
	}
    
// fprintf(stderr,"minloc: %d(%f) at %dth iter!\n",minloc,minval,iter);
// fprintf(stderr," /* %04d */ ",iter);
// fprintf(stderr,"%f,",(features->data.fl+iter*features->cols)[0]);
// fprintf(stderr,"%.0f,",(features->data.fl+iter*features->cols)[1]);
// fprintf(stderr,"%f,",(features->data.fl+iter*features->cols)[2]);
// cvPrintf(stderr,"%.0f,",features,cvRect(3,iter,features->cols-3,1));

#if 1
	static CvWindowManager winmgr;
    // display min err feature
    {
      CvMat * disp = cvCreateMat(tsize,tsize,CV_8U);
      cvSet(disp,cvScalar(128));
      // memcpy(disp->data.ptr,posimgs[0]->data.ptr,(tsize-1)*(tsize-1));
      float * curptr = features->data.fl+minloc*features->cols+3;
      CvMat subdisp0_stub,subdisp1_stub;
      CvMat * subdisp0 = cvGetSubRect(disp,&subdisp0_stub,
                         cvRect(curptr[0],curptr[1],curptr[2],curptr[3]));
      CvMat * subdisp1 = cvGetSubRect(disp,&subdisp1_stub,
                         cvRect(curptr[5],curptr[6],curptr[7],curptr[8]));
	  if ((features->data.fl+minloc*features->cols)[1]>0){
		cvSet(subdisp0,cvScalar((curptr[4]>0)?255:0));
		cvSet(subdisp1,cvScalar((curptr[9]>0)?255:0));
	  }else{
		cvSet(subdisp0,cvScalar((curptr[4]>0)?0:255));
		cvSet(subdisp1,cvScalar((curptr[9]>0)?0:255));
	  }
      CV_SHOW(disp);
      cvReleaseMat(&disp);
    }
#endif

    cvReleaseMat(&epsilon);
// break;
  } // end of iter loop

  // release memory
  for (i=0;i<2;i++){cvReleaseMat(&evalres[i]);}
  // cvReleaseMat(&weights);
  // cvReleaseMat(&features);
  for (i=0;i<2;i++){
    for (j=0;j<count[i];j++){ cvReleaseMat(&imgIntegral[i][j]); }
    delete [] imgIntegral[i];
  }

  return 1;
}

// validate i-th classifier on the validation set
int CvStagedDetectorHaar::validate(int ni, double & fi, double & di)
{
#if CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
  int i,j,k,iter;float thres,polar; int fcc=0,dcc=0;
  int count[2];
  for (i=0;i<2;i++) { count[i] = evalres_precomp[i]->cols; }
  CvMat * frate = cvCreateMat(1,count[0],CV_32S);
  CvMat * drate = cvCreateMat(1,count[1],CV_32S);
  cvSet(frate,cvScalar(1));
  cvSet(drate,cvScalar(1));
  int * frateptr = frate->data.i;
  int * drateptr = drate->data.i;
  for (k=0;k<selected.size();k++){
  iter=selected[k];
  thres=CV_MAT_ELEM(*features,float,iter,0);
  polar=CV_MAT_ELEM(*features,float,iter,1);
  for (i=0;i<2;i++){
  float * evalresptr=
      (float*)(evalres_precomp[i]->data.ptr+evalres_precomp[i]->step*iter);
  for (j=0;j<evalres_precomp[i]->cols;j++){
    if (evalresptr[j]*polar<thres*polar) {
    }else{
      if (i==0) {frateptr[j]=0;}  // false positive
      if (i==1) {drateptr[j]=0;}  // detection ratio
    }
  } // j loop
  } // i loop
  } // k loop
  fi=cvSum(frate).val[0]/double(count[0]);
  di=cvSum(drate).val[0]/double(count[1]);
  cvReleaseMat(&frate);
  cvReleaseMat(&drate);
#else
#error "not implemented!"
#endif // CV_STAGED_DETECT_HAAR_PRECOMPUTE_EVAL
  return 1;
}

// adjust i-th threshold to target detection ratio
int CvStagedDetectorHaar::adjust(int ni, double dtar,
                                 double & fi, double & di)
{
  int i,j,iter=selected[selected.size()-1];float thres,polar; int fcc=0,dcc=0;
  float * featptr = (float*)(features->data.ptr+features->step*iter);
  float * evalresptr;
  if (featptr[1]>0){
    while(di<dtar){
      featptr[0]+=1;
fcc=0;dcc=0;
thres=CV_MAT_ELEM(*features,float,iter,0);
polar=CV_MAT_ELEM(*features,float,iter,1);
for (i=0;i<2;i++){
evalresptr=evalres_precomp[i]->data.fl+evalres_precomp[i]->cols*iter;
for (j=0;j<evalres_precomp[i]->cols;j++){
if ((evalresptr[j]*polar<thres*polar)&&(i==0)) {fcc++;}
if ((evalresptr[j]*polar<thres*polar)&&(i==1)) {dcc++;}
}
}
fi=double(fcc)/double(evalres_precomp[0]->cols);
di=double(dcc)/double(evalres_precomp[1]->cols);
    }
  }else{
    while(di<dtar){
      featptr[0]-=1;
fcc=0;dcc=0;
thres=CV_MAT_ELEM(*features,float,iter,0);
polar=CV_MAT_ELEM(*features,float,iter,1);
for (i=0;i<2;i++){
evalresptr=evalres_precomp[i]->data.fl+evalres_precomp[i]->cols*iter;
for (j=0;j<evalres_precomp[i]->cols;j++){
if ((evalresptr[j]*polar<thres*polar)&&(i==0)) {fcc++;}
if ((evalresptr[j]*polar<thres*polar)&&(i==1)) {dcc++;}
}
}
fi=double(fcc)/double(evalres_precomp[0]->cols);
di=double(dcc)/double(evalres_precomp[1]->cols);
    }
  }
  // fprintf(stderr,"ni:%d,thres:%f,di:%f\n",ni,featptr[0],di);
  return 1;
}

// cascade detector training framework 
int CvStagedDetectorHaar::
cascadetrain(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
             double fper, double dper, double ftarget)
{
  int i,ii,j,k,ni,maxiter=200; double fi,di;
  CvMat * Frate = cvCreateMat(1,maxiter,CV_64F); Frate->data.db[0]=1.0;
  CvMat * Drate = cvCreateMat(1,maxiter,CV_64F); Drate->data.db[0]=1.0;
  CvMat * stage = cvCreateMat(1,maxiter,CV_32S); stage->data.i[0]=0;

  for (i=0,ni=0;(Frate->data.db[i]>ftarget)&&(i<maxiter);)
  {
    i++;
    fi=1.0;
    for (;fi>fper*Frate->data.db[i-1];){
      // fprintf(stderr,"%f>%f\n",fi,fper*Frate->data.db[i-1]);
      ni++;

      // adaboost training
      train(posimgs,npos,negimgs,nneg,ni,ni-1);

      // validation
      validate(ni,fi,di);

      // adjust threshold for i-th classifier
      // adjust(ni,dper*Drate->data.db[i-1],fi,di);
      adjust(ni,dper,fi,di);

// update weights
// if (0)      
// {
//   int nfeatures=features->rows;
//   int count[2]={evalres_precomp[0]->cols,evalres_precomp[1]->cols};
//   int nsamples = count[0]+count[1];
//   CvMat * epsilon = cvCreateMat(1,nfeatures,CV_64F);
//   cvSet(epsilon,cvScalar(-1));
//   int iter=ni-1;double epsval;
//   float thres = CV_MAT_ELEM(*features,float,iter,0);
//   float polar = CV_MAT_ELEM(*features,float,iter,1);
//   double * wtptr;
//   for (ii=iter;ii<nfeatures;ii++){
//     wtptr = weights->data.db+nsamples*iter;
//     epsval=0;
//     for (j=0;j<2;j++){
//       float * evalresptr =
//           evalres_precomp[j]->data.fl+evalres_precomp[j]->cols*ii;
//       double * wt0ptr = wtptr+count[0]*j;
//       for (k=0;k<count[j];k++){
//         epsval+=wt0ptr[k]*
//             fabs((((evalresptr[k]*polar)<(thres*polar))?1:0)-j);
//       }
//     }
//     epsilon->data.db[ii]=epsval;
//   }
//   double minval=0xffffff; int minloc;
//   for (ii=iter;ii<nfeatures;ii++){
//     epsval = epsilon->data.db[ii];
//     if (epsval<minval){minval=epsval;minloc=ii;}
//   }
//   double * wt0ptr = weights->data.db+nsamples*iter;
//   double * wt1ptr = weights->data.db+nsamples*(iter+1);
//   // float thres = (features->data.fl+iter*features->cols)[0];
//   // float polar = (features->data.fl+iter*features->cols)[1];
//   double beta = minval/(1.-minval),ei;
//   (features->data.fl+iter*features->cols)[2]=log(1./(beta+1e-6));
//   k=0;
//   for (ii=0;ii<2;ii++){
//   // float * evalresptr = evalres[i]->data.fl;
//   float * evalresptr =
//       evalres_precomp[ii]->data.fl+evalres_precomp[ii]->cols*iter;
//   for (j=0;j<count[ii];j++,k++){
//     ei = ((evalresptr[j]*polar)<(thres*polar))?1:0;
//     wt1ptr[k]=wt0ptr[k]*pow(beta,ei);
//   }
//   }
//   cvReleaseMat(&epsilon);
// }

      // compute fi under this threshold
      validate(ni,fi,di);
      //if (fi==fi_old){ni--;}
fprintf(stderr,"/* %d,fi:%.4f,di:%.2f */",selected[selected.size()-1],fi,di);
fprintf(stderr,"%.2f,",(features->data.fl+(selected[selected.size()-1])*features->cols)[0]);
fprintf(stderr,"%.0f,",(features->data.fl+(selected[selected.size()-1])*features->cols)[1]);
fprintf(stderr,"%.2f,",(features->data.fl+(selected[selected.size()-1])*features->cols)[2]);
cvPrintf(stderr,"%.0f,",features,cvRect(3,selected[selected.size()-1],features->cols-3,1));
    }
    Frate->data.db[i]=fi;//Frate->data.db[i-1]*fi;
    Drate->data.db[i]=di;//Drate->data.db[i-1];
    stage->data.i[i]=ni;

    // add to negative training set ...
    fprintf(stderr,"// end of %d-th stage with %d classifiers\n",i,ni);
  }

  {
    fprintf(stderr,"/* %d stages: */\n//",i);
    for (ii=0;ii<i+1;ii++){fprintf(stderr,"%d,",stage->data.i[ii]);}
  }
  
  cvReleaseMat(&Frate);
  cvReleaseMat(&Drate);
  cvReleaseMat(&stage);
  
  return 1;
}

CvMat * icvCreateHaarFeatureSetEx(int tsize)
{
  typedef float haar[18]; assert(sizeof(haar)==18*4);
  int log2count=8;
  haar * buff = (haar*)malloc(sizeof(haar)*(1<<log2count));
  int i,j,m,n,count=0,pstep=2,x,y,dx,dy;

  for (x=0;x<tsize;x+=pstep){
  for (y=0;y<tsize;y+=pstep){
  // for (x=0;x<tsize;x++){
  // for (y=0;y<tsize;y++){
  for (dx=2;dx<tsize;dx+=pstep){
  for (dy=2;dy<tsize;dy+=pstep){

  if ((1<<log2count)-10<count){
    buff = (haar*)realloc(buff,sizeof(haar)*(1<<(++log2count)));
    if (!buff) {
      fprintf(stderr, "ERROR: memory allocation failure!\n"); exit(1);
    }
  }

  if ((x+dx*2<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,x,y,dx*2,dy,-1,x+dx,y,dx,dy,2,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*2<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,y,x,dy,dx*2,-1,y,x+dx,dy,dx,2,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*3<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,x,y,dx*3,dy,-1,x+dx,y,dx,dy,3,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*3<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,y,x,dy,dx*3,-1,y,x+dx,dy,dx,3,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*4<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,x,y,dx*4,dy,-1,x+dx,y,dx*2,dy,2,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*4<tsize)&&(y+dy<tsize)){
    float tmp[18]={0,0,0,y,x,dy,dx*4,-1,y,x+dx,dy,dx*2,2,0,0,0,0,0};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  if ((x+dx*2<tsize)&&(y+dy*2<tsize)){
    float tmp[18]={0,0,0,x,y,dx*2,dy*2,-1,x,y,dx,dy,2,x+dx,y+dy,dx,dy,2};
    memcpy(buff[count++],tmp,sizeof(haar));
  }
  
  }
  }
  }
  }

  CvMat * features = cvCreateMat(count,18,CV_32F);
  memcpy(features->data.ptr,buff,sizeof(haar)*count);
  free(buff);
  return features;
}

