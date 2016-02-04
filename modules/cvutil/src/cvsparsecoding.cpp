/**
 * @file   cvsparsecoding.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Jul 19 16:30:06 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvsparsecoding.h"
#include "cvtimer.h"

int icvLog2(double x){
  static double invlog2 = 1./log(2.);
  return cvRound(log(x)*invlog2);
}

#ifndef ANDROID
int CvSparseLearner::learn(const char * filelist[], int nfiles)
{
  int i,j,k;
  const int dictsize=8;
  const int dictshiftbits=icvLog2(dictsize);
  for (i=0;i<nfiles;i++){
    IplImage * raw = cvLoadImage(filelist[0],0);
    CvMat img_stub;
    CvMat * img = cvGetMat(raw,&img_stub);
    int nr=(img->rows>>dictshiftbits),nc=(img->cols>>dictshiftbits);
    int N=nr*nc;
    int M=dictsize*dictsize;
    CvMat * dict = cvCreateMat(N,M,CV_8U);
    CvMat * signal = cvCreateMat(N,M,CV_32F);
    CvMat * D = cvCreateMat(N,M,CV_32F);

    icvDictDenseSample(img,dict);
    cvConvert(dict,signal);
    icvDictShuffle(dict);
    cvConvert(dict,D);

    CvMat * y = cvCreateMat(M,1,CV_32F);
    CvMat * x = cvCreateMat(1,N,CV_32F);
    // for (j=0;;j++)
    // {
    //   float patch_data[64];
    //   CvMat patch = cvMat(8,8,CV_32F,patch_data);
    //   // memcpy(patch_data,D->data.fl+249*M,sizeof(float)*M);
    //   memcpy(patch_data,D->data.fl+j*M,sizeof(float)*M);
    //   //cvPrintf(stderr,"%.2f,",&patch);
    //   CvMat * resized = cvCreateMat(128,128,CV_32F);
    //   cvResize(&patch,resized);
    //   CV_SHOW(resized);
    //   cvReleaseMat(&resized);
    // }
    
    for (j=0;j<N;j++){
      memcpy(y->data.fl,signal->data.fl+j*M,sizeof(float)*M);
      icvOrthogonalMatchingPursuit(D,y,x,100,5.);
      for (k=0;k<N;k++){
        if (x->data.fl[k]!=0)fprintf(stderr,"%.2f,",x->data.fl[k]);
      }fprintf(stderr,"\n");
      CV_SHOW(img);
    }
    cvReleaseMat(&x);
    cvReleaseMat(&y);

    CV_SHOW(img);
    icvDictShow(dict); CV_WAIT();

    cvReleaseMat(&D);
    cvReleaseMat(&dict);
    cvReleaseImage(&raw);
  }

  return 1;
}
#endif //ANDROID

int icvDictDenseSample(CvMat * img, CvMat * dict)
{
  const int dictsize=sqrt(float(dict->cols)); 
  double aa=log(2.);double bb=log(8.);double cc=bb/aa;
  const int shiftbits=icvLog2(dictsize); assert(shiftbits==3);
  int j,k,i;
  int nr=(img->rows>>shiftbits),nc=(img->cols>>shiftbits);
  int N=nr*nc; 
  int M=dictsize*dictsize;
  assert(dict->rows==N);
  assert(dict->cols==M);
  assert(CV_MAT_TYPE(img->type)==CV_8U);
  assert(CV_MAT_TYPE(dict->type)==CV_8U);

  int count;
  assert(CV_MAT_TYPE(img->type)==CV_8U);
  int step=img->step;
  fprintf(stderr,"dict: %d,%d\n",N,M);

  // collect all image patches
  for (j=0;j<nr;j++){
    for (k=0;k<nc;k++){
      for (i=0;i<dictsize;i++){
        memcpy(dict->data.ptr+(j*nc+k)*M+(i<<shiftbits),
               img->data.ptr+((j<<shiftbits)+i)*step+(k<<shiftbits),
               dictsize);
      }
    }
  }

  return 1;
}

int icvDictShuffle(CvMat * dict)
{
  int N=dict->rows,M=dict->cols,j;
  CvMat * perm = cvCreateMat(N,1,CV_32S);
  for (j=0;j<N;j++){perm->data.i[j]=j;}
  cvRandShuffle(perm,0);
  CvMat * buff = cvCloneMat(dict);

  for (j=0;j<N;j++){
    memcpy(dict->data.ptr+M*j,buff->data.ptr+M*perm->data.i[j],M);
  }

  cvReleaseMat(&buff);
  cvReleaseMat(&perm);

  return 1;
}

#ifndef ANDROID
int icvDictShow(CvMat * dict)
{
  int i,j,k;
  int N=dict->rows;
  int M=dict->cols;
  int dictsize=sqrt(float(M));
  int shiftbits=icvLog2(dictsize);
  assert((1<<shiftbits)==dictsize);
  int nc=cvFloor(sqrt(float(N)));
  int nr=cvFloor(N/nc);
  int type = CV_MAT_TYPE(dict->type);
  int step;
  CvMat * disp=0;
  disp=cvCreateMat(nr*dictsize,nc*dictsize,type);cvZero(disp);
  if (type==CV_8U){
    step = disp->step;
    for (j=0;j<nr;j++){
      for (k=0;k<nc;k++){
        for (i=0;i<dictsize;i++){
          memcpy(disp->data.ptr+((j<<shiftbits)+i)*step+(k<<shiftbits),
                 dict->data.ptr+(j*nc+k)*M+(i<<shiftbits),dictsize);
        }
      }
    }
  }else if (type==CV_32F){
    step = disp->step/sizeof(float);
    for (j=0;j<nr;j++){
      for (k=0;k<nc;k++){
        for (i=0;i<dictsize;i++){
          memcpy(disp->data.fl+((j<<shiftbits)+i)*step+(k<<shiftbits),
                 dict->data.fl+(j*nc+k)*M+(i<<shiftbits),
                 dictsize*sizeof(float));
        }
      }
    }
  }else{
    fprintf(stderr,"ERROR: function unimplemented for the data type!\n");
  }
  if (disp){
    cvShowImageEx("Test",disp,CV_CM_GRAY);
    cvReleaseMat(&disp);
  }
  return 1;
}
#endif // ANDROID

int icvOrthogonalMatchingPursuit(CvMat * D, CvMat * y, CvMat * x,
                                 int maxiter, float epsilon)
{
  assert(D&&y);
  int N=D->rows; // number of dictionary elements
  int M=D->cols; // length of signal
  assert(y->rows==M);
  assert(CV_MAT_TYPE(D->type)==CV_32F);
  assert(CV_MAT_TYPE(y->type)==CV_32F);
  assert(CV_MAT_TYPE(x->type)==CV_32F);
  CvMat * err = cvCloneMat(y);
  int i,j,iter=0,converged=0;
  CvMat * inner = cvCreateMat(1,N,CV_32F);
  CvMat * phi = cvCreateMat(maxiter,M,CV_32F); cvZero(phi);
  int * loc = new int[maxiter];
  double minval,maxval; CvPoint minloc,maxloc;
  
  cvZero(x);

  for (iter=0;;iter++){
    CvMat * subphi = cvCreateMat(M,iter+1,CV_32F);
    CvMat * val = cvCreateMat(iter+1,1,CV_32F);
    CvMat * a = cvCreateMat(subphi->rows,val->cols,CV_32F);

    // inner product
    cvGEMM(err,D,1,NULL,1,inner,CV_GEMM_A_T+CV_GEMM_B_T);
    cvMinMaxLoc(inner,&minval,&maxval,&minloc,&maxloc);
    loc[iter]=maxloc.x;

    memcpy(phi->data.fl+iter*M,D->data.fl+loc[iter]*M,sizeof(float)*M);
    for (i=0;i<M;i++){
    for (j=0;j<subphi->cols;j++){
      CV_MAT_ELEM(*subphi,float,i,j)=CV_MAT_ELEM(*phi,float,j,i);
    }
    }
    cvSolve(subphi,y,val,CV_SVD); // least square solution

    cvMatMul(subphi,val,a);
    cvSub(y,a,err);

    if ((cvNorm(err,0,CV_L2,0)<epsilon)||(iter+1>=maxiter)) { // l2-norm 
      // for (i=0;i<iter+1;i++){
      //   fprintf(stderr,"%d:%.2f,",loc[i],val->data.fl[i]);}
      for (i=0;i<iter+1;i++){x->data.fl[loc[i]]=val->data.fl[i];}
      converged=1;
    } 
    cvReleaseMat(&a);
    cvReleaseMat(&subphi);
    cvReleaseMat(&val);
    if (converged){break;}
  }//fprintf(stderr, "\n");
  
  delete loc;
  cvReleaseMat(&phi);
  cvReleaseMat(&inner);
  cvReleaseMat(&err);
  return 1;
}

int icvBasisPursuit(CvMat * A, CvMat * b, CvMat * x, int maxiter)
{
  int iter,i;
  int M=A->rows,N=A->cols;
  assert((M==b->rows)&&(b->cols==1));
  assert((N==x->rows)&&(1==x->cols));
  cvSet(x,cvScalar(.1));
  CvMat * c = cvCreateMat(N,1,CV_32F);     cvSet(c,cvScalar(1));
  CvMat * z = cvCreateMat(N,1,CV_32F);     cvSet(z,cvScalar(.1));
  CvMat * y = cvCreateMat(M,1,CV_32F);     cvZero(y);
  CvMat * mu = cvCreateMat(N,1,CV_32F);    cvSet(mu,cvScalar(.1));
  CvMat * e = cvCreateMat(N,1,CV_32F);     cvSet(e,cvScalar(1));
  CvMat * eyeN = cvCreateMat(N,N,CV_32F);  cvSetIdentity(eyeN,cvScalar(1));
  CvMat * eyeM = cvCreateMat(M,M,CV_32F);  cvSetIdentity(eyeM,cvScalar(1));
  CvMat * z_diag = cvCreateMat(N,N,CV_32F);
  CvMat * x_diag = cvCreateMat(N,N,CV_32F);
  CvMat * x_diag_inv = cvCreateMat(N,N,CV_32F);
  const double gamma=1e-4,delta=1e-4;
  const double sqrgamma=gamma*gamma,sqrdelta=delta*delta;
  double rho_p,rho_d;

  CvMat * t = cvCreateMat(N,1,CV_32F);
  CvMat * r = cvCreateMat(M,1,CV_32F);
  CvMat * v = cvCreateMat(N,1,CV_32F);
  CvMat * D = cvCreateMat(N,N,CV_32F);
  CvMat * dx = cvCreateMat(N,1,CV_32F);
  CvMat * dy = cvCreateMat(M,1,CV_32F);
  CvMat * dz = cvCreateMat(N,1,CV_32F);

  CvMat * tmpMxN = cvCreateMat(M,N,CV_32F);
  CvMat * tmpMxM = cvCreateMat(M,M,CV_32F);
  CvMat * tmpNx1 = cvCreateMat(N,1,CV_32F);
  CvMat * tmpMx1 = cvCreateMat(M,1,CV_32F);
  
  //CvMat * c,z,x,mu,e,eyeN,eyeM,z_diag,x_diag,x_diag_inv,t,r,v,D,dx,dy,dz,
  //tmpMxN,tmpMxM,tmpNx1,tmpMx1
      
  for (iter=0;iter<maxiter;iter++){
    icvDiag(z,z_diag);
    icvDiag(x,x_diag);
    
    //-------------------------------------------------------
    //  (a) compute residuals and diagonal matrix D
    //------------------------------------------------------- 
    // t=c+(gamma^2)*x-z-A'*y;
    cvGEMM(A,y,-1,x,sqrgamma,t,CV_GEMM_A_T);
    cvAdd(t,c,t);cvSub(t,z,t);
    // r=b-A*x-(delta^2)*y;
    cvGEMM(A,x,-1,y,-sqrdelta,r,0);
    cvAdd(r,b,r);
    // v=mu.*e-diag(z)*x;
    cvMul(mu,e,v);
    cvGEMM(z_diag,x,-1,v,1,v,0);
    // D=inv(inv(diag(x))*diag(z)+(gamma^2)*I);
    if (0==cvInvert(x_diag,x_diag_inv,CV_LU)){cvInvert(x_diag,x_diag_inv,CV_SVD);}
    cvGEMM(x_diag_inv,z_diag,1,eyeN,sqrgamma,D,0);
    if (0==cvInvert(D,D,CV_LU)){cvInvert(D,D,CV_SVD);}

    //-------------------------------------------------------
    //  (b) solve least square approximation
    //------------------------------------------------------- 
    // dy=(A*D*A'+(delta^2)*eye(length(b)))\(r+A*D*(t-inv(diag(x))*v));
    cvMatMul(A,D,tmpMxN);
    cvGEMM(tmpMxN,A,1,eyeM,sqrdelta,tmpMxM,CV_GEMM_B_T);
    cvMatMul(x_diag_inv,v,tmpNx1);
    cvSub(t,tmpNx1,tmpNx1);
    cvGEMM(tmpMxN,tmpNx1,1,r,1,tmpMx1,0);
    cvSolve(tmpMxM,tmpMx1,dy,CV_SVD);
    // dx=D*(A'*dy+inv(diag(x))*v-t);
    cvGEMM(A,dy,1,t,-1,tmpNx1,CV_GEMM_A_T);
    cvGEMM(x_diag_inv,v,1,tmpNx1,1,tmpNx1,0);
    cvMatMul(D,tmpNx1,dx);
    // dz=inv(diag(x))*(v-diag(z)*dx);
    cvMatMul(z_diag,dx,tmpNx1);
    cvSub(v,tmpNx1,tmpNx1);
    cvMatMul(x_diag_inv,tmpNx1,dz);

    //-------------------------------------------------------
    //  (c) update the variables
    //-------------------------------------------------------
    rho_p=1<<20;rho_d=1<<20;
    {
      int negdxcount=0,negdzcount=0;
      float * dxptr=dx->data.fl;
      float * dzptr=dz->data.fl;
      for (i=0;i<N;i++){
        if (dxptr[i]<0){negdxcount+=1;}
        if (dzptr[i]<0){negdzcount+=1;}
      }
      if (negdxcount){
        float * xptr = x->data.fl;
        for (i=0;i<N;i++){
          if (dxptr[i]<0){ rho_p = MIN(rho_p,-xptr[i]/dxptr[i]); }
        }
      }
      if (negdzcount){
        float * zptr = z->data.fl;
        for (i=0;i<N;i++){
          if (dzptr[i]<0){ rho_d = MIN(rho_d,-zptr[i]/dzptr[i]); }
        }
      }
      rho_p=MIN(.99*rho_p,1.);
      rho_d=MIN(.99*rho_d,1.);
    }
    
    cvScaleAdd(dx,cvScalar(rho_p),x,x); 
    cvScaleAdd(dy,cvScalar(rho_d),y,y);
    cvScaleAdd(dz,cvScalar(rho_d),z,z);
    {
      double scale=1.-MIN(rho_p,MIN(rho_d,.99));
      float * muptr=mu->data.fl;
      for (i=0;i<N;i++){muptr[i]=muptr[i]*scale;}
    }

    //-------------------------------------------------------
    //  :-) termination criteria
    //-------------------------------------------------------
    // gap=(z'*x)/(1+norm(z)*norm(x));
    double normx=cvNorm(x);
    double gap=cvDotProduct(z,x)/(1.+cvNorm(z)+normx);
    if ((cvNorm(r)/(1.+normx)<.1)&&             // primal infeasibility
        (cvNorm(t)/(1+cvNorm(y))<.1)&&          // dual infeasibility
        (gap<.1))                               // duality gap
    {break;}
  }
  
  // release following variables:
  //    c,z,x,mu,e,eyeN,eyeM,
  ///   z_diag,x_diag,x_diag_inv,t,r,v,D,dx,dy,dz,
  //    tmpMxN,tmpMxM,tmpNx1,tmpMx1
  cvReleaseMat(&c);
  cvReleaseMat(&z);
  cvReleaseMat(&y);
  cvReleaseMat(&mu);
  cvReleaseMat(&e);
  cvReleaseMat(&eyeM);
  cvReleaseMat(&eyeN);

  cvReleaseMat(&z_diag);
  cvReleaseMat(&x_diag);
  cvReleaseMat(&x_diag_inv);

  cvReleaseMat(&t);
  cvReleaseMat(&r);
  cvReleaseMat(&v);
  cvReleaseMat(&D);

  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&dz);

  cvReleaseMat(&tmpMxN);
  cvReleaseMat(&tmpMxM);
  cvReleaseMat(&tmpMx1);
  cvReleaseMat(&tmpNx1);

  return 1;
}

int icvBasisPursuit_optimized(CvMat * A, CvMat * b, CvMat * x)
{
  int i;
  int M=A->rows,N=A->cols;
  assert((M==b->rows)&&(b->cols==1));
  assert((N==x->rows)&&(1==x->cols));
  cvSet(x,cvScalar(.1));
  CvMat * c = cvCreateMat(N,1,CV_32F);     cvSet(c,cvScalar(1));
  CvMat * z = cvCreateMat(N,1,CV_32F);     cvSet(z,cvScalar(.1));
  CvMat * y = cvCreateMat(M,1,CV_32F);     cvZero(y);
  CvMat * mu = cvCreateMat(N,1,CV_32F);    cvSet(mu,cvScalar(.1));
  CvMat * e = cvCreateMat(N,1,CV_32F);     cvSet(e,cvScalar(1));
  CvMat * eyeN = cvCreateMat(N,N,CV_32F);  cvSetIdentity(eyeN,cvScalar(1));
  CvMat * eyeM = cvCreateMat(M,M,CV_32F);  cvSetIdentity(eyeM,cvScalar(1));
  CvMat * z_diag = cvCreateMat(N,N,CV_32F);
  CvMat * x_diag = cvCreateMat(N,N,CV_32F);
  CvMat * x_diag_inv = cvCreateMat(N,N,CV_32F);
  const double gamma=1e-4,delta=1e-4;
  const double sqrgamma=gamma*gamma,sqrdelta=delta*delta;
  double rho_p,rho_d;

  CvMat * t = cvCreateMat(N,1,CV_32F);
  CvMat * r = cvCreateMat(M,1,CV_32F);
  CvMat * v = cvCreateMat(N,1,CV_32F);
  CvMat * D = cvCreateMat(N,N,CV_32F);
  CvMat * dx = cvCreateMat(N,1,CV_32F);
  CvMat * dy = cvCreateMat(M,1,CV_32F);
  CvMat * dz = cvCreateMat(N,1,CV_32F);

  CvMat * tmpMxN = cvCreateMat(M,N,CV_32F);
  CvMat * tmpMxM = cvCreateMat(M,M,CV_32F);
  CvMat * tmpNx1 = cvCreateMat(N,1,CV_32F);
  CvMat * tmpMx1 = cvCreateMat(M,1,CV_32F);
  
  //CvMat * c,z,x,mu,e,eyeN,eyeM,z_diag,x_diag,x_diag_inv,t,r,v,D,dx,dy,dz,
  //tmpMxN,tmpMxM,tmpNx1,tmpMx1
      
  {
    icvDiag(z,z_diag);
    icvDiag(x,x_diag);
    
    //-------------------------------------------------------
    //  (a) compute residuals and diagonal matrix D
    //------------------------------------------------------- 
    // t=c+(gamma^2)*x-z-A'*y;
    cvGEMM(A,y,-1,x,sqrgamma,t,CV_GEMM_A_T);
    cvAdd(t,c,t);cvSub(t,z,t);
    // r=b-A*x-(delta^2)*y;
    cvGEMM(A,x,-1,y,-sqrdelta,r,0);
    cvAdd(r,b,r);
    // v=mu.*e-diag(z)*x;
    cvMul(mu,e,v);
    cvGEMM(z_diag,x,-1,v,1,v,0);
    // D=inv(inv(diag(x))*diag(z)+(gamma^2)*I);
    if (0==cvInvert(x_diag,x_diag_inv,CV_LU)){cvInvert(x_diag,x_diag_inv,CV_SVD);}
    cvGEMM(x_diag_inv,z_diag,1,eyeN,sqrgamma,D,0);
    if (0==cvInvert(D,D,CV_LU)){cvInvert(D,D,CV_SVD);}

    //-------------------------------------------------------
    //  (b) solve least square approximation
    //------------------------------------------------------- 
    // dy=(A*D*A'+(delta^2)*eye(length(b)))\(r+A*D*(t-inv(diag(x))*v));
    cvMatMul(A,D,tmpMxN);
    cvGEMM(tmpMxN,A,1,eyeM,sqrdelta,tmpMxM,CV_GEMM_B_T);
    cvMatMul(x_diag_inv,v,tmpNx1);
    cvSub(t,tmpNx1,tmpNx1);
    cvGEMM(tmpMxN,tmpNx1,1,r,1,tmpMx1,0);
    cvSolve(tmpMxM,tmpMx1,dy,CV_SVD);

    //-------------------------------------------------------
    //  (c) update the variables
    //-------------------------------------------------------
    rho_p=1<<20;
    {
      int negdxcount=0;
      float * dxptr=dx->data.fl;
      for (i=0;i<N;i++){
        if (dxptr[i]<0){negdxcount+=1;}
      }
      if (negdxcount){
        float * xptr = x->data.fl;
        for (i=0;i<N;i++){
          if (dxptr[i]<0){ rho_p = MIN(rho_p,-xptr[i]/dxptr[i]); }
        }
      }
      rho_p=MIN(.99*rho_p,1.);
    }
    
    cvScaleAdd(dx,cvScalar(rho_p),x,x); 
  }
  
  // release following variables:
  //    c,z,x,mu,e,eyeN,eyeM,
  ///   z_diag,x_diag,x_diag_inv,t,r,v,D,dx,dy,dz,
  //    tmpMxN,tmpMxM,tmpNx1,tmpMx1
  cvReleaseMat(&c);
  cvReleaseMat(&z);
  cvReleaseMat(&y);
  cvReleaseMat(&mu);
  cvReleaseMat(&e);
  cvReleaseMat(&eyeM);
  cvReleaseMat(&eyeN);

  cvReleaseMat(&z_diag);
  cvReleaseMat(&x_diag);
  cvReleaseMat(&x_diag_inv);

  cvReleaseMat(&t);
  cvReleaseMat(&r);
  cvReleaseMat(&v);
  cvReleaseMat(&D);

  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&dz);

  cvReleaseMat(&tmpMxN);
  cvReleaseMat(&tmpMxM);
  cvReleaseMat(&tmpMx1);
  cvReleaseMat(&tmpNx1);

  return 1;
}
