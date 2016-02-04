/**
 * @file   cvimgwarp.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Feb 20 14:00:32 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvimgwarp.h"

#define __CV_IMG_WARP_TYPE_32F_FIELDS__             \
  assert( CV_MAT_TYPE(img->type)==CV_32F );         \
  assert( CV_MAT_TYPE(dst->type)==CV_32F );         \
  assert( (warp_p->cols==1) );                      \
  float * iptr = img->data.fl;                      \
  float * wptr = dst->data.fl;                      \
  float * pptr = warp_p->data.fl;                   \
  int y,x,nr=dst->rows,nc=dst->cols,                \
    istep=img->step/sizeof(float),                  \
    wstep=dst->step/sizeof(float);

#define __CV_IMG_WARP_TYPE_8U_FIELDS__               \
  assert( CV_MAT_TYPE(img->type)==CV_8U );           \
  assert( CV_MAT_TYPE(dst->type)==CV_8U );           \
  assert( (warp_p->cols==1) );                       \
  uchar * iptr = img->data.ptr;                      \
  uchar * wptr = dst->data.ptr;                      \
  float * pptr = warp_p->data.fl;                    \
  int y,x,nr=dst->rows,nc=dst->cols,                 \
    istep=img->step/sizeof(uchar),                   \
    wstep=dst->step/sizeof(uchar);

#define __CV_IMG_WARP_FOR_LOOP_FIELDS__                      \
  for (y=0;y<nr;y++,wptr+=wstep)                             \
  {                                                          \
    for (x=0;x<nc;x++)                                       \
    {                                                        \
      xpos = cvRound(x*cp0+y*sp1+pptr[xx]);                  \
      ypos = cvRound(x*sp3+y*cp4+pptr[yy]);                  \
      if ( (xpos>ww) || (xpos<0) || (ypos>hh) || (ypos<0) )  \
      {                                                      \
        wptr[x]=1e-5;                                        \
      }else{                                                 \
        wptr[x] = (iptr+istep*ypos)[xpos];                   \
      }                                                      \
    }                                                        \
  }                                                         
  

void icvWarp0_32f(CvMat * img, CvMat * dst, CvMat * warp_p);
void icvWarp0_8u (CvMat * img, CvMat * dst, CvMat * warp_p);
void icvWarp3_32f(CvMat * src, CvMat * dst, CvMat * warp_p);
void icvWarp3_8u(CvMat * src, CvMat * dst, CvMat * warp_p);
void icvWarp4_32f(CvMat * src, CvMat * dst, CvMat * warp_p);
void icvWarp4_8u(CvMat * src, CvMat * dst, CvMat * warp_p);
void icvWarp6_32f(CvMat * src, CvMat * dst, CvMat * warp_p);
void icvWarp6_8u(CvMat * src, CvMat * dst, CvMat * warp_p);

CV_INLINE
void icvWarpInit0(CvMat * warp_p, float tx, float ty)
{
  warp_p->data.fl[0]=1.f;
  warp_p->data.fl[1]=0.f;
  warp_p->data.fl[2]=tx;//5.*m_iLinMultiplier;
  warp_p->data.fl[3]=0.f;
  warp_p->data.fl[4]=1.f;
  warp_p->data.fl[5]=ty;//5.*m_iLinMultiplier;
}

CV_INLINE
void icvWarpInit3(CvMat * warp_p, float tx, float ty)
{
  warp_p->data.fl[0]=1.f;
  warp_p->data.fl[1]=tx;//5.*m_iLinMultiplier;
  warp_p->data.fl[2]=ty;//5.*m_iLinMultiplier;
}

CV_INLINE
void icvWarpInit4(CvMat * warp_p, float tx, float ty)
{
  warp_p->data.fl[0]=1.f;
  warp_p->data.fl[1]=0.f;
  warp_p->data.fl[2]=tx;//5.*m_iLinMultiplier;
  warp_p->data.fl[3]=ty;//5.*m_iLinMultiplier;
}

CV_INLINE
void icvWarpInit6(CvMat * warp_p, float tx, float ty)
{
  warp_p->data.fl[0]=1.f;
  warp_p->data.fl[1]=0.f;
  warp_p->data.fl[2]=0.f;
  warp_p->data.fl[3]=1.f;
  warp_p->data.fl[4]=tx;//5.*m_iLinMultiplier;
  warp_p->data.fl[5]=ty;//5.*m_iLinMultiplier;
}

CVAPI(void) icvWarpInit(CvMat * warp_p, float tx, float ty)
{
  typedef void (*CvWarpInitFuncType)(CvMat *, float, float);
  static CvWarpInitFuncType warpinitfuncarr[7]={
    &icvWarpInit0,0,0,&icvWarpInit3,&icvWarpInit4,0,&icvWarpInit6
  };
  warpinitfuncarr[(warp_p->cols==1)?warp_p->rows:0](warp_p,tx,ty);
}

CV_INLINE
void icvWarpInvert0(CvMat * warp_p, CvMat * invwarp_p3x3)
{
  assert(CV_MAT_TYPE(warp_p->type)==CV_32F);
  CvMat * warp_p3x3 = cvCreateMat(3,3,CV_32F);
  memcpy(warp_p3x3->data.fl, warp_p->data.fl, sizeof(float)*6);
  warp_p3x3->data.fl[6]=0;
  warp_p3x3->data.fl[7]=0;
  warp_p3x3->data.fl[8]=1.;
  if (cvInvert(warp_p3x3, invwarp_p3x3, CV_LU)==0.0){
    cvInvert(warp_p3x3, invwarp_p3x3, CV_SVD);
  }
  cvReleaseMat(&warp_p3x3);
}

CV_INLINE
void icvWarpInvert3(CvMat * warp_p, CvMat * invwarp_p3x3)
{
  assert(warp_p->rows==3);
  assert(CV_MAT_TYPE(warp_p->type)==CV_32F);
  CvMat * warp_p3x3 = cvCreateMat(3,3,CV_32F);
  float * p33ptr = warp_p3x3->data.fl;
  float * pptr = warp_p->data.fl;
  p33ptr[0]=pptr[0]; // scale = scale*cos(0)
  p33ptr[1]=0;
  p33ptr[2]=pptr[1];
  p33ptr[3]=0;
  p33ptr[4]=pptr[0];
  p33ptr[5]=pptr[2];
  p33ptr[6]=0;
  p33ptr[7]=0;
  p33ptr[8]=1;
  if (cvInvert(warp_p3x3, invwarp_p3x3, CV_LU)==0.0){
    cvInvert(warp_p3x3, invwarp_p3x3, CV_SVD);
  }
  cvReleaseMat(&warp_p3x3);
}

CV_INLINE
void icvWarpInvert4(CvMat * warp_p, CvMat * invwarp_p3x3)
{
  assert(warp_p->rows==4);
  assert(CV_MAT_TYPE(warp_p->type)==CV_32F);
  CvMat * warp_p3x3 = cvCreateMat(3,3,CV_32F);
  float * p33ptr = warp_p3x3->data.fl;
  float * pptr = warp_p->data.fl;
  p33ptr[0]=pptr[0];
  p33ptr[1]=-pptr[1];
  p33ptr[2]=pptr[2];
  p33ptr[3]=pptr[1];
  p33ptr[4]=pptr[0];
  p33ptr[5]=pptr[3];
  p33ptr[6]=0;
  p33ptr[7]=0;
  p33ptr[8]=1;
  if (cvInvert(warp_p3x3, invwarp_p3x3, CV_LU)==0.0){
    cvInvert(warp_p3x3, invwarp_p3x3, CV_SVD);
  }
  cvReleaseMat(&warp_p3x3);
}

CV_INLINE
void icvWarpInvert6(CvMat * warp_p, CvMat * invwarp_p3x3)
{
  assert(CV_MAT_TYPE(warp_p->type)==CV_32F);
  CvMat * warp_p3x3 = cvCreateMat(3,3,CV_32F);
  float * p33ptr = warp_p3x3->data.fl;
  float * pptr = warp_p->data.fl;
  p33ptr[0]=pptr[0];
  p33ptr[1]=pptr[2];
  p33ptr[2]=pptr[4];
  p33ptr[3]=pptr[1];
  p33ptr[4]=pptr[3];
  p33ptr[5]=pptr[5];
  p33ptr[6]=0;
  p33ptr[7]=0;
  p33ptr[8]=1;
  if (cvInvert(warp_p3x3, invwarp_p3x3, CV_LU)==0.0){
    cvInvert(warp_p3x3, invwarp_p3x3, CV_SVD);
  }
  cvReleaseMat(&warp_p3x3);
}

CVAPI(void) icvWarpInvert(CvMat * warp_p, CvMat * invwarp_p3x3)
{
  typedef void (*CvWarpInvertFuncType)(CvMat *, CvMat *);
  static CvWarpInvertFuncType warpinvfuncarr[7]={
    &icvWarpInvert0,0,0,&icvWarpInvert3,&icvWarpInvert4,0,&icvWarpInvert6
  };
  warpinvfuncarr[(warp_p->cols==1)?warp_p->rows:0](warp_p,invwarp_p3x3);
}

/** 
 * warp current frame to template size, using nearest-neighbor method
 * 
 * @param img       in:  input floating point image, assumed to be CV_32F
 * @param dst      out: output warped template image, assumed to be CV_32F
 * @param warp_p    in:  2x3 matrix defined as [R t],
 *                       represent for rotation and translation
 */
CVAPI(void) icvWarp(CvMat * img, CvMat * dst,
                    CvMat * warp_p)
{
  typedef void (*CvWarpFuncType)(CvMat *,CvMat *,CvMat *);
  static CvWarpFuncType warpfuncarr[14] = {
    0,0,0,&icvWarp3_32f,&icvWarp4_32f,0,&icvWarp6_32f,
    0,0,0,&icvWarp3_8u ,&icvWarp4_8u ,0,&icvWarp6_8u 
  };
#if defined(__x86_64__) || defined(_WIN64) || defined(__ppc64__)
  //#error "possible error here !!!"
  static CvMat warpfuncmat = cvMat(2,7,CV_64F,warpfuncarr);
#else
  static CvMat warpfuncmat = cvMat(2,7,CV_32S,warpfuncarr);
#endif
  // static CvMat warpfuncmat = cvMat(2,7,CV_32S,warpfuncarr);
  int srctype = CV_MAT_TYPE(img->type);
  int dsttype = CV_MAT_TYPE(dst->type);
  int pnr=warp_p->rows;
  int pnc=warp_p->cols;
  if ( ((pnr==2)||(pnr==3)) && (pnc==3) )
    // 2x3 matrix OR 3x3 matrix
  {
    if ( (srctype==CV_32F) && (dsttype==CV_32F) )
    {
      icvWarp0_32f(img, dst, warp_p);
    }else if ( (srctype==CV_8U) )
    {
      // if (dsttype==CV_32F) { icvWarp_8u_32f(img, dst, warp_p); }
      // else
      if ( (dsttype==CV_8U) ) { icvWarp0_8u(img, dst, warp_p); }
      else {assert(false);}
    }else{
      assert(false);
    }
  }else if (pnc==1){
    assert( (warp_p->rows==3) || (warp_p->rows==4) || (warp_p->rows==6) );
    //warpfuncarr[warp_p->rows](img, dst, warp_p);
    CV_MAT_ELEM(warpfuncmat, CvWarpFuncType, (srctype==CV_8U), warp_p->rows)
        (img, dst, warp_p);
  }else{
    assert(false);
  }
}

void icvWarp0_32f(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(img->type)==CV_32F );
  assert( CV_MAT_TYPE(dst->type)==CV_32F );
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  assert( (warp_p->cols==3) );
  float * iptr = img->data.fl;
  float * wptr = dst->data.fl;
  float * pptr = warp_p->data.fl;
  int y,x,nr=dst->rows,nc=dst->cols,
      istep=img->step/sizeof(float),
      wstep=dst->step/sizeof(float);
  int xpos, ypos;
  float cp0=pptr[0], sp1=pptr[1], sp3=pptr[3], cp4=pptr[4];
  int ww=img->width-1, hh=img->height-1;
  for (y=0;y<nr;y++,wptr+=wstep)
  {
    for (x=0;x<nc;x++)
    {
      xpos = cvRound(x*cp0+y*sp1+pptr[2]); // nearest-neighbor method
      ypos = cvRound(x*sp3+y*cp4+pptr[5]);
      if ( (xpos>ww) || (xpos<0) || (ypos>hh) || (ypos<0) )
      {
        wptr[x]=1e-5;
      }else{
        wptr[x] = (iptr+istep*ypos)[xpos];
      }
    }
  }
}

void icvWarp0_8u(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(img->type)==CV_8U );
  assert( CV_MAT_TYPE(dst->type)==CV_8U );
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  assert( (warp_p->cols==3) );
  uchar * iptr = img->data.ptr;
  uchar * wptr = dst->data.ptr;
  float * pptr = warp_p->data.fl;
  int y,x,nr=dst->rows,nc=dst->cols,
      istep=img->step/sizeof(uchar),
      wstep=dst->step/sizeof(uchar);
  int xpos, ypos;
  float cp0=pptr[0], sp1=pptr[1], sp3=pptr[3], cp4=pptr[4];
  int ww=img->width-1, hh=img->height-1;
  for (y=0;y<nr;y++,wptr+=wstep)
  {
    for (x=0;x<nc;x++)
    {
      xpos = cvRound(x*cp0+y*sp1+pptr[2]); // nearest-neighbor method
      ypos = cvRound(x*sp3+y*cp4+pptr[5]);
      if ( (xpos>ww) || (xpos<0) || (ypos>hh) || (ypos<0) )
      {
        wptr[x]=0;
      }else{
        wptr[x]=(iptr+istep*ypos)[xpos];
      }
    }
  }
}

void icvWarp3_8u(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_8U_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=0, sp3=0, cp4=pptr[0];
  int xx=1,yy=2;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

void icvWarp3_32f(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_32F_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=0, sp3=0, cp4=pptr[0];
  int xx=1,yy=2;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

void icvWarp4_32f(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_32F_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  int xx=2,yy=3;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

void icvWarp4_8u(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_8U_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  int xx=2,yy=3;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

void icvWarp6_32f(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_32F_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=pptr[2], sp3=pptr[1], cp4=pptr[3];
  int xx=4,yy=5;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

void icvWarp6_8u(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  __CV_IMG_WARP_TYPE_8U_FIELDS__
  int xpos, ypos;
  float cp0=pptr[0], sp1=pptr[2], sp3=pptr[1], cp4=pptr[3];
  int xx=4,yy=5;
  int ww=img->width-1, hh=img->height-1;
  __CV_IMG_WARP_FOR_LOOP_FIELDS__
}

double icvWarpToPoints3(CvMat * warp_p,
                        CvPoint2D32f pts[4], int hh, int ww)
{
  assert( (warp_p->rows==3) || (warp_p->cols==1));
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  float * pptr = warp_p->data.fl;
  float cp0=pptr[0], sp1=0, sp3=0, cp4=pptr[0];
  int xx=1,yy=2;
  pts[0].x = pptr[xx]; // nearest-neighbor method
  pts[0].y = pptr[yy];
  pts[1].x = ww*cp0+pptr[xx]; // nearest-neighbor method
  pts[1].y = ww*sp3+pptr[yy];
  pts[2].x = ww*cp0+hh*sp1+pptr[xx]; // nearest-neighbor method
  pts[2].y = ww*sp3+hh*cp4+pptr[yy];
  pts[3].x = hh*sp1+pptr[xx]; // nearest-neighbor method
  pts[3].y = hh*cp4+pptr[yy];
  return 0;//acos(cp0/sqrt(cp0*cp0+sp1*sp1))*180./CV_PI;;
}

double icvWarpToPoints4(CvMat * warp_p,
                        CvPoint2D32f pts[4], int hh, int ww)
{
  assert( (warp_p->rows==4) || (warp_p->cols==1));
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  float * pptr = warp_p->data.fl;
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  int xx=2,yy=3;
  pts[0].x = pptr[xx]; // nearest-neighbor method
  pts[0].y = pptr[yy];
  pts[1].x = cp0*ww+pptr[xx]; // nearest-neighbor method
  pts[1].y = sp1*ww+pptr[yy];
  pts[2].x = cp0*ww+sp1*hh+pptr[xx]; // nearest-neighbor method
  pts[2].y = sp3*ww+cp4*hh+pptr[yy];
  pts[3].x = sp1*hh+pptr[xx]; // nearest-neighbor method
  pts[3].y = cp0*hh+pptr[yy];
  // return acos(cp0/sqrt(cp0*cp0+sp1*sp1))*180./CV_PI;
  return atan2(sp1,cp0)*180./CV_PI;
  // return asin(sp1/sqrt(cp0*cp0+sp1*sp1))*180./CV_PI;
}

double icvWarpToPoints6(CvMat * warp_p,
                        CvPoint2D32f pts[4], int hh, int ww)
{
  assert( (warp_p->rows==6) || (warp_p->cols==1));
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  float * pptr = warp_p->data.fl;
  float cp0=pptr[0], sp1=pptr[2], sp3=pptr[1], cp4=pptr[3];
  int xx=4,yy=5;
  pts[0].x = pptr[xx]; // nearest-neighbor method
  pts[0].y = pptr[yy];
  pts[1].x = ww*cp0+pptr[xx]; // nearest-neighbor method
  pts[1].y = ww*sp3+pptr[yy];
  pts[2].x = ww*cp0+hh*sp1+pptr[xx]; // nearest-neighbor method
  pts[2].y = ww*sp3+hh*cp4+pptr[yy];
  pts[3].x = hh*sp1+pptr[xx]; // nearest-neighbor method
  pts[3].y = hh*cp4+pptr[yy];
  // return acos(cp0/sqrt(cp0*cp0+sp1*sp1))*180./CV_PI;;
  return asin(sp1/sqrt(cp0*cp0+sp1*sp1))*180./CV_PI;;
}

CVAPI(double) icvWarpToPoints(CvMat * warp_p,
                              CvPoint2D32f pts[4], int hh, int ww)
{
  assert(warp_p->cols==1); // column vector
  typedef double (*CvWarpToPointsFuncType)
      (CvMat *, CvPoint2D32f pts[4], int, int);
  static CvWarpToPointsFuncType warptopfuncarr[7] = {
    0,0,0,&icvWarpToPoints3,&icvWarpToPoints4,0,&icvWarpToPoints6
  };
  return warptopfuncarr[warp_p->rows](warp_p,pts,hh,ww);
}

CV_INLINE
void icvWarpCompose3(CvMat * comp_M, CvMat * warp_p)
{
  assert( (warp_p->cols==1) || (warp_p->rows==3) );
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  assert( CV_MAT_TYPE(comp_M->type)==CV_32F );
  float * pptr = warp_p->data.fl;
  float * cptr = comp_M->data.fl;
  pptr[0]=cptr[0]-1.;
  pptr[1]=cptr[2];
  pptr[2]=cptr[5];
}

CV_INLINE
void icvWarpCompose4(CvMat * comp_M, CvMat * warp_p)
{
  assert( (warp_p->cols==1) || (warp_p->rows==4) );
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  assert( CV_MAT_TYPE(comp_M->type)==CV_32F );
  float * pptr = warp_p->data.fl;
  float * cptr = comp_M->data.fl;
  pptr[0]=cptr[0]-1.;
  pptr[1]=cptr[3];
  pptr[2]=cptr[2];
  pptr[3]=cptr[5];
}

CV_INLINE
void icvWarpCompose6(CvMat * comp_M, CvMat * warp_p)
{
  assert( (warp_p->cols==1) || (warp_p->rows==6) );
  float * pptr = warp_p->data.fl;
  float * cptr = comp_M->data.fl;
  pptr[0]=cptr[0]-1.;
  pptr[1]=cptr[3];
  pptr[2]=cptr[1];
  pptr[3]=cptr[4]-1.;
  pptr[4]=cptr[2];
  pptr[5]=cptr[5];
}

CVAPI(void) icvWarpCompose(CvMat * comp_M, CvMat * warp_p)
{
  assert(warp_p->cols==1);
  typedef void (*CvWarpCompseFuncType)(CvMat *, CvMat *);
  static CvWarpCompseFuncType warpcompfuncarr[7] = {
    0,0,0,&icvWarpCompose3,&icvWarpCompose4,0,&icvWarpCompose6
  };
  warpcompfuncarr[warp_p->rows](comp_M, warp_p);
}

void icvWarpTranspose3(CvMat * delta_p, CvMat * delta_M)
{
  assert(delta_p->rows==3);
  delta_M->data.fl[0]=delta_p->data.db[0]+1.; // diagonal
  delta_M->data.fl[1]=0;
  delta_M->data.fl[2]=delta_p->data.db[1];
  delta_M->data.fl[3]=0;
  delta_M->data.fl[4]=delta_p->data.db[0]+1.; // diagonal element
  delta_M->data.fl[5]=delta_p->data.db[2];
  delta_M->data.fl[6]=0;
  delta_M->data.fl[7]=0;
  delta_M->data.fl[8]=1.;
}

void icvWarpTranspose4(CvMat * delta_p, CvMat * delta_M)
{
  assert(delta_p->rows==4);
  delta_M->data.fl[0]=delta_p->data.db[0]+1.; // diagonal
  delta_M->data.fl[1]=delta_p->data.db[1];
  delta_M->data.fl[2]=delta_p->data.db[2];
  delta_M->data.fl[3]=delta_p->data.db[1];
  delta_M->data.fl[4]=delta_p->data.db[0]+1.; // diagonal element
  delta_M->data.fl[5]=delta_p->data.db[3];
  delta_M->data.fl[6]=0;
  delta_M->data.fl[7]=0;
  delta_M->data.fl[8]=1.;
}

void icvWarpTranspose6(CvMat * delta_p, CvMat * delta_M)
{
  assert(delta_p->rows==6);
  delta_M->data.fl[0]=delta_p->data.db[0]+1.; // diagonal
  delta_M->data.fl[1]=delta_p->data.db[2];
  delta_M->data.fl[2]=delta_p->data.db[4];
  delta_M->data.fl[3]=delta_p->data.db[1];
  delta_M->data.fl[4]=delta_p->data.db[3]+1.; // diagonal element
  delta_M->data.fl[5]=delta_p->data.db[5];
  delta_M->data.fl[6]=0;
  delta_M->data.fl[7]=0;
  delta_M->data.fl[8]=1.;
}

CVAPI(void) icvWarpTranspose(CvMat * delta_p, CvMat * delta_M)
{
  assert(delta_p->cols==1);
  typedef void (*CvWarpTransposeFuncType)(CvMat *, CvMat *);
  static CvWarpTransposeFuncType warptransposefuncarr[7] = {
    0,0,0,&icvWarpTranspose3,&icvWarpTranspose4,0,&icvWarpTranspose6
  };
  warptransposefuncarr[delta_p->rows](delta_p,delta_M);
}

void icvWarpReshape3(CvMat * warp_p, CvMat * warp_M)
{
  assert( (warp_p->cols==1) || (warp_p->rows==3) );
  float * Mptr = warp_M->data.fl;
  float * pptr = warp_p->data.fl;
  Mptr[0]=pptr[0]+1.;
  Mptr[1]=0;
  Mptr[2]=pptr[1];
  Mptr[3]=0;
  Mptr[4]=pptr[0]+1.;
  Mptr[5]=pptr[2];
  Mptr[6]=0;
  Mptr[7]=0;
  Mptr[8]=1.;
}

void icvWarpReshape4(CvMat * warp_p, CvMat * warp_M)
{
  assert( (warp_p->cols==1) || (warp_p->rows==4) );
  float * Mptr = warp_M->data.fl;
  float * pptr = warp_p->data.fl;
  Mptr[0]=pptr[0]+1.;
  Mptr[1]=-pptr[1];
  Mptr[2]=pptr[2];
  Mptr[3]=pptr[1];
  Mptr[4]=pptr[0]+1.;
  Mptr[5]=pptr[3];
  Mptr[6]=0;
  Mptr[7]=0;
  Mptr[8]=1.;
}

void icvWarpReshape6(CvMat * warp_p, CvMat * warp_M)
{
  assert( (warp_p->cols==1) || (warp_p->rows==6) );
  float * Mptr = warp_M->data.fl;
  float * pptr = warp_p->data.fl;
  Mptr[0]=pptr[0]+1.;
  Mptr[1]=pptr[2];
  Mptr[2]=pptr[4];
  Mptr[3]=pptr[1];
  Mptr[4]=pptr[3]+1.;
  Mptr[5]=pptr[5];
  Mptr[6]=0;
  Mptr[7]=0;
  Mptr[8]=1.;
}

CVAPI(void) icvWarpReshape(CvMat * warp_p, CvMat * warp_M)
{
  typedef void (*CvWarpReshapeFuncType)(CvMat *, CvMat *);
  static CvWarpReshapeFuncType warpreshapefuncarr[7] = {
    0,0,0,&icvWarpReshape3,&icvWarpReshape4,0,&icvWarpReshape6
  };
  warpreshapefuncarr[warp_p->rows](warp_p, warp_M);
}

