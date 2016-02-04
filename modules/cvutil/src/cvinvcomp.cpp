/**
 * @file   cvinvcomp.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Sat Feb 16 14:25:40 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvinvcomp.h"

void icvCalcJacobian3(CvMat * dW_dp, const int nr, const int nc)
{
  assert(dW_dp->rows==nr*2);
  assert(dW_dp->cols==nc*3);
  int x,y;
  int step=dW_dp->cols;
  assert(CV_MAT_TYPE(dW_dp->type)==CV_32F);
  assert(step==dW_dp->step/sizeof(float));
  float * dptr00 = dW_dp->data.fl;
  float * dptr10 = dW_dp->data.fl + step*nr;
  float * dptr01 = dW_dp->data.fl +           nc;
  float * dptr12 = dW_dp->data.fl + step*nr + nc*2;
  cvZero(dW_dp);
  for (y=0;y<nr;y++){
    for (x=0;x<nc;x++){
      dptr00[x]=x;  dptr10[x]=y;
      dptr01[x]=1;  dptr12[x]=1; 
    }
    dptr00+=step; dptr10+=step;
    dptr01+=step; dptr12+=step;
  }
}

void icvCalcJacobian4(CvMat * dW_dp, const int nr, const int nc)
{
  assert(dW_dp->rows==nr*2);
  assert(dW_dp->cols==nc*4);
  int x,y;
  int step=dW_dp->cols;
  assert(CV_MAT_TYPE(dW_dp->type)==CV_32F);
  assert(step==dW_dp->step/sizeof(float));
  float * dptr00 = dW_dp->data.fl;
  float * dptr10 = dW_dp->data.fl + step*nr;
  float * dptr01 = dW_dp->data.fl +           nc;
  float * dptr11 = dW_dp->data.fl + step*nr + nc;
  float * dptr02 = dW_dp->data.fl +           nc*2;
  float * dptr13 = dW_dp->data.fl + step*nr + nc*3;
  cvZero(dW_dp);
  for (y=0;y<nr;y++){
    for (x=0;x<nc;x++){
      dptr00[x]=x;  dptr10[x]=y;
      dptr01[x]=-y; dptr11[x]=x;
      dptr02[x]=1;  dptr13[x]=1; 
    }
    dptr00+=step; dptr01+=step;
    dptr10+=step; dptr11+=step;
    dptr02+=step; dptr13+=step;
  }
}

void icvCalcJacobian6(CvMat * dW_dp, const int nr, const int nc)
{
  assert(dW_dp->rows==nr*2);
  assert(dW_dp->cols==nc*6);
  int i,j;
  int step=dW_dp->cols;
  float * dxptr00 = dW_dp->data.fl;
  float * dxptr11 = dW_dp->data.fl + step*nr + nc;
  float * dyptr02 = dW_dp->data.fl +           nc*2;
  float * dyptr13 = dW_dp->data.fl + step*nr + nc*3;
  float * dzptr04 = dW_dp->data.fl +           nc*4;
  float * dzptr15 = dW_dp->data.fl + step*nr + nc*5;
  cvZero(dW_dp);
  for (i=0;i<nr;i++){
    for (j=0;j<nc;j++){
      dxptr00[j]=j; dxptr11[j]=j;
      dyptr02[j]=i; dyptr13[j]=i;
      dzptr04[j]=1; dzptr15[j]=1; 
    }
    dxptr00+=step; dxptr11+=step;
    dyptr02+=step; dyptr13+=step;
    dzptr04+=step; dzptr15+=step;
  }
}

/** 
 * calculate jacobian matrix for affine warp
 * 
 * @param dW_dp   out: 2x6 jacobian matrix for affine warp
 * @param nr      in:  number of rows of template image
 * @param nc      in:  number of cols of template image
 */
CVAPI(void) icvCalcJacobian(CvMat * dW_dp, const int nr, const int nc)
{
  int N_p = dW_dp->cols/nc;
  assert( (N_p==3) || (N_p==4) || (N_p==6));
  typedef void (*CvCalcJacobFuncType)(CvMat *, const int, const int);
  static CvCalcJacobFuncType jacobfuncarr[7] = {
    0,0,0,&icvCalcJacobian3,&icvCalcJacobian4,0,&icvCalcJacobian6
  };
  jacobfuncarr[N_p](dW_dp, nr, nc);
}

/** 
 * 
 * 
 * @param dW_dp 
 * @param nabla_Ix 
 * @param nabla_Iy 
 * @param VI_dW_dp 
 * @param N_p 
 */
CVAPI(void) icvCalcStDescImages(CvMat * dW_dp,
                                CvMat * nabla_Ix, CvMat * nabla_Iy,
                                CvMat * VI_dW_dp)
{
  int p,i,j,nr=nabla_Ix->rows,nc=nabla_Ix->cols;
  const int N_p = VI_dW_dp->cols/nc; //assert(N_p==6);
  int dststep = VI_dW_dp->cols;
  int dpstep = dW_dp->cols;
  int step = nc;
  float * dxptr, * dyptr;
  float * dpptr0, * dpptr1, * dstptr; 
  for (p=0;p<N_p;p++)
  {
    dxptr = nabla_Ix->data.fl;
    dyptr = nabla_Iy->data.fl;
    dpptr0 = dW_dp->data.fl +             nc*p;
    dpptr1 = dW_dp->data.fl + dpstep*nr + nc*p;
    dstptr = VI_dW_dp->data.fl + nc*p; 
    for (i=0;i<nr;i++){
      for (j=0;j<nc;j++){
        dstptr[j]=dxptr[j]*dpptr0[j]+dyptr[j]*dpptr1[j];
      }
      dpptr0+=dpstep; dpptr1+=dpstep; 
      dxptr+=step; dyptr+=step; dstptr+=dststep;
    }
  }
}

/** 
 * calculate 6x6 hessian matrix from 1x6 steepest descent images
 * 
 * @param VI_dW_dp in:  steepest descent images, assumed to be CV_32F
 * @param H        out: hessian matrix, assumed to be CV_64F
 */
CVAPI(void) icvCalcHessian(CvMat * VI_dW_dp, CvMat * H)
{
  assert(CV_MAT_TYPE(VI_dW_dp->type)==CV_32F);
  assert(CV_MAT_TYPE(H->type)==CV_64F);
  cvZero(H);
  int i,j,m,n,N_p=H->rows,
      dnr=VI_dW_dp->rows,dnc=VI_dW_dp->cols,
      nc=VI_dW_dp->cols/H->cols;
  double * hptr = H->data.db;
  float * h1ptr, * h2ptr;
  int hstep = H->step/sizeof(double);
  int dstep = VI_dW_dp->step/sizeof(float);
  assert( H->cols==N_p );
  assert( hstep==N_p );
  for (i=0;i<N_p;i++)
  {
    for (j=0;j<N_p;j++)
    {
      h1ptr = VI_dW_dp->data.fl+nc*i;
      h2ptr = VI_dW_dp->data.fl+nc*j;
      for (m=0;m<dnr;m++){
        for (n=0;n<nc;n++){
          hptr[j]+=h1ptr[n]*h2ptr[n];
        }
        h1ptr+=dstep;h2ptr+=dstep;
      }
    }
    hptr+=hstep;
  }
}

/** 
 * multiply sd_image with error_img to get sd parameters
 * 
 * @param VI_dW_dp    in:  steepest descent image,
 *                         the multiplication of gradient and jacobian
 * @param error_img   in:  error image = gradient - template
 * @param sd_delta_p  out: 6x1 matrix steepest descent 
 */
CVAPI(void) icvUpdateStDescImages(CvMat * VI_dW_dp, CvMat * error_img,
                                  CvMat * sd_delta_p)
{
#if 1
  cvZero(sd_delta_p);
  int N_p=sd_delta_p->rows;
  int nr=VI_dW_dp->rows, nc=VI_dW_dp->cols/N_p;
  int i,j,k;
  float * eptr, * dptr;
  double * sdptr = sd_delta_p->data.db;
  int estep = error_img->step/sizeof(float); 
  int dstep = VI_dW_dp->step/sizeof(float);
  assert(estep*N_p==dstep);
  assert(estep==error_img->cols);
  assert(dstep==VI_dW_dp->cols);
  assert(nc==error_img->cols);
  for (i=0;i<N_p;i++) // for each element in sd_delta_p
  {
    dptr = VI_dW_dp->data.fl+nc*i;
    eptr = error_img->data.fl;
    for (j=0;j<nr;j++){
      for (k=0;k<nc;k++){
        sdptr[i] += dptr[k]*eptr[k];
      }
      eptr+=estep; dptr+=dstep;
    }
  }
#else
  int N_p=sd_delta_p->rows;
  int i,nr=error_img->rows,nc=error_img->cols;
  CvMat * tmp = cvCreateMat(nr,nc,CV_32F);
  for (i=0;i<N_p;i++){
    CvMat * mat, mat_stub;
    mat = cvGetSubRect(VI_dW_dp, &mat_stub,
                       cvRect(nc*i,0,nc,nr));
    cvMul(mat, error_img, tmp);
    sd_delta_p->data.db[i]=cvSum(tmp).val[0];
  }
  cvReleaseMat(&tmp);
#endif
}
