/**
 * @file   cvinvcomp.h
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Dec 24 16:10:43 2012
 * 
 * @brief  
 *
 * sample USAGE:
 *
 #include "highgui.h"
 #include "cvinvcomp.h"

 int main(int argc, char * argv[])
 {
 // IplImage * img = cvLoadImage("../data/takeo.pgm", 0);
 IplImage * img2 = cvLoadImage("../data/111.png", 0);
 IplImage * img3 = cvLoadImage("../data/222.png", 0);
 CvMat mat_stub, * mat, submat_stub, * submat, * init_p;
 CvMat * mat32f, * submat32f;
 const int maxiter=200;
 CvInvComp invcomp;

 mat = cvCloneMat(cvGetMat(img3, &mat_stub));
 // submat = cvCloneMat(cvGetSubRect(mat, &submat_stub,
 //                                  cvRect(50, 90, 50, 50)));
 submat = cvCloneMat(cvGetSubRect(img2, &submat_stub,
 cvRect(61, 120, 36, 42)));

 mat32f = cvCreateMat(mat->rows, mat->cols, CV_32F);
 submat32f = cvCreateMat(submat->rows, submat->cols, CV_32F);

 cvConvert(mat, mat32f);
 cvConvert(submat, submat32f);

 init_p = cvCreateMat(2, 3, CV_32F); 
 init_p->data.fl[0]=1;
 init_p->data.fl[1]=0;
 // init_p->data.fl[2]=55;
 // init_p->data.fl[3]=0;
 // init_p->data.fl[4]=1;
 // init_p->data.fl[5]=95;
 init_p->data.fl[2]=55;
 init_p->data.fl[3]=0;
 init_p->data.fl[4]=1;
 init_p->data.fl[5]=120;

 #if 1
 invcomp.affine_fa(mat32f, submat32f, init_p, maxiter, 600);
 #else
 invcomp.affine_ic(mat32f, submat32f, init_p, maxiter, 600);
 #endif

 cvReleaseMat(&mat);
 cvReleaseMat(&submat);
  
 return 0;
 }
 * 
 */

#include "cxcore.h"
#include "cv.h"
#include "cvext_c.h"
#include "cvtracker.h"

// extern void printminmax(CvMat *);

class CvInvComp
{
  CvMat * img_dx;
  CvMat * img_dy;
  CvMat * dW_dp;
  CvMat * IWxp;
  CvMat * warp_p;

  CvMat * error_img;
  CvMat * nabla_Ix;
  CvMat * nabla_Iy;
  CvMat * VI_dW_dp;
  CvMat * H;
  CvMat * H_inv;
  CvMat * sd_delta_p;
  CvMat * delta_p;

  CvMat * delta_M;
  CvMat * delta_M_inv;
  CvMat * warp_M;
  CvMat * comp_M;

 public:
  CvInvComp():
      img_dx(NULL), img_dy(NULL), dW_dp(NULL),
      IWxp(NULL), warp_p(NULL),
      error_img(NULL),
      nabla_Ix(NULL), nabla_Iy(NULL),
      VI_dW_dp(NULL), H(NULL), H_inv(NULL),
      sd_delta_p(NULL), delta_p(NULL),
      delta_M(NULL), delta_M_inv(NULL), warp_M(NULL), comp_M(NULL)
  {
    
  }

  ~CvInvComp()
  {
    cvReleaseMat(&img_dx);
    cvReleaseMat(&img_dy); 
    cvReleaseMat(&dW_dp); 
    cvReleaseMat(&IWxp); 
    cvReleaseMat(&warp_p); 
    cvReleaseMat(&error_img);
    cvReleaseMat(&nabla_Ix);
    cvReleaseMat(&nabla_Iy);
    cvReleaseMat(&VI_dW_dp);
    cvReleaseMat(&H);
    cvReleaseMat(&H_inv);
    cvReleaseMat(&sd_delta_p);
    cvReleaseMat(&delta_p);
    cvReleaseMat(&delta_M);
    cvReleaseMat(&delta_M_inv);
    cvReleaseMat(&warp_M);
    cvReleaseMat(&comp_M);
  }

  void init_a(CvMat * img, CvMat * tmplt, CvMat * p_init,
              const int N_p, const int nr, const int nc)
  {
    assert( (p_init->rows==2) && (p_init->cols==3) ); // column vector
    assert( CV_MAT_TYPE(img->type)==CV_32F );
    assert( CV_MAT_TYPE(tmplt->type)==CV_32F );
    
    if (!dW_dp) { dW_dp = cvCreateMat(nr*2,nc*6,CV_32F); }
    if (!IWxp) { IWxp = cvCreateMat(nr,nc,CV_32F); }
    if (!warp_p) { warp_p = cvCreateMat(2, 3, CV_32F); }
    if (!error_img) { error_img = cvCreateMat(nr, nc, CV_32F); }
    if (!nabla_Ix) { nabla_Ix = cvCreateMat(nr, nc, CV_32F); }
    if (!nabla_Iy) { nabla_Iy = cvCreateMat(nr, nc, CV_32F); }
    if (!VI_dW_dp) { VI_dW_dp = cvCreateMat(nr, nc*N_p, CV_32F); }
    if (!H) { H = cvCreateMat(N_p, N_p, CV_64F); }          // double
    if (!H_inv) { H_inv = cvCreateMat(N_p, N_p, CV_64F); }  // double
    if (!sd_delta_p) { sd_delta_p = cvCreateMat(N_p, 1, CV_64F); } // double
    if (!delta_p) { delta_p = cvCreateMat(6, 1, CV_64F); } // double 
    if (!delta_M) { delta_M = cvCreateMat(3, 3, CV_32F); }
    if (!delta_M_inv) { delta_M_inv = cvCreateMat(3, 3, CV_32F); }
    if (!warp_M) { warp_M = cvCreateMat(3, 3, CV_32F); }
    if (!comp_M) { comp_M = cvCreateMat(3, 3, CV_32F); }

    cvCopy(p_init, warp_p);
  }

  void affine_ic(CvMat * img, CvMat * tmplt, CvMat * p_init,
                 const int maxiter, const double eps)
  {
    int nr=tmplt->rows,nc=tmplt->cols,iter;
    const int N_p=6;

    if (!img_dx) { img_dx = cvCreateMat(nr, nc, CV_32F); }
    if (!img_dy) { img_dy = cvCreateMat(nr, nc, CV_32F); }
    init_a(img, tmplt, p_init, N_p, nr, nc);

    // step 3: evaluate the gradient of the template
    cvSobel(tmplt, img_dx, 1, 0, 1); cvScale(img_dx, img_dx, 0.5);
    cvSobel(tmplt, img_dy, 0, 1, 1); cvScale(img_dy, img_dy, 0.5);

    // step 4: evaluate the Jacobian dW_dp
    jacobian_a(dW_dp, nr, nc);

    // step 5: compute the steepest descent images
    sd_images(dW_dp, img_dx, img_dy, VI_dW_dp, N_p);
// cvShowImage("Test", VI_dW_dp); CV_WAIT();
// CV_SHOW(VI_dW_dp);
    
    // step 6: compute the Hessian matrix
    hessian(VI_dW_dp, H);
    if (cvInvert(H, H_inv, CV_LU)==0.0) {cvInvert(H, H_inv, CV_SVD);}
// cvPrintf(stderr, "%g ", H);
// cvPrintf(stderr, "%g ", H_inv);
// cvShowImageEx("Test", H, CV_CM_GRAY); CV_WAIT();
// cvShowImageEx("Test", H_inv, CV_CM_GRAY); CV_WAIT();

    double rms_error=0;
    // float prewarp_p_data[6];
    // CvMat prewarp_p = cvMat(2,3,CV_32F,prewarp_p_data);

// cvShowImageEx("Test", tmplt, CV_CM_GRAY); CV_WAIT();
// CV_TIMER_START();
    for (iter=0;iter<maxiter;iter++)
    {
      // step 1: warp I with W(x;p) to compute I(W(x;p))
      warp_a(img, IWxp, warp_p);
// cvShowImageEx("Test", IWxp, CV_CM_GRAY); CV_WAIT();
      
      // step 2: compute the error image I(W(x;p))-T(x)
      rms_error=0;
      {
        float * tptr = tmplt->data.fl;
        float * wptr = IWxp->data.fl;
        float * eptr = error_img->data.fl;
        int tstep=tmplt->step/sizeof(float);
        int wstep=IWxp->step/sizeof(float);
        int estep=error_img->step/sizeof(float);
        int i,j;
        for (i=0;i<nr;i++){
          for (j=0;j<nc;j++){
            eptr[j]=tptr[j]-wptr[j];
            rms_error += eptr[j]*eptr[j];
          }
          tptr+=tstep; wptr+=wstep; eptr+=estep;
        }
        rms_error = sqrt(rms_error);
      }
// CV_SHOW(error_img);
// fprintf(stderr, "rms_error: %f\n\n", rms_error);
      if (rms_error<eps) {break;}

      // step 7: update steepest descent images
      sd_update(VI_dW_dp, error_img, sd_delta_p);
// fprintf(stderr, "sd_delta_p:\n");
// cvPrintf(stderr, "%f ", sd_delta_p);

      // step 8: compute delta_p
      // cvMatMul(H_inv, sd_delta_p, delta_p);
      cvZero(delta_p);
      {
        double * hptr=H_inv->data.db;
        int i,j;
        for (i=0;i<N_p;i++,hptr+=H_inv->cols) {
          for (j=0;j<N_p;j++) {
            delta_p->data.db[i] += hptr[j]*sd_delta_p->data.db[j];
          }
        }
      }
// fprintf(stderr, "delta_p:\n");
// cvPrintf(stderr, "%f ", delta_p);

      // step 9: update the warp 
      {
        delta_M->data.fl[0]=delta_p->data.db[0]+1.; // diagonal
        delta_M->data.fl[1]=delta_p->data.db[2];
        delta_M->data.fl[2]=delta_p->data.db[4];
        delta_M->data.fl[3]=delta_p->data.db[1];
        delta_M->data.fl[4]=delta_p->data.db[3]+1.; // diagonal element
        delta_M->data.fl[5]=delta_p->data.db[5];
        delta_M->data.fl[6]=0;
        delta_M->data.fl[7]=0;
        delta_M->data.fl[8]=1.;

        // invert compositional warp
        if (cvInvert(delta_M, delta_M_inv, CV_LU)==0.0) {
          cvInvert(delta_M, delta_M_inv, CV_SVD);
        }

        // current warp
        memcpy(warp_M->data.fl, warp_p->data.fl, sizeof(float)*6);
        warp_M->data.fl[0]+=1.;
        warp_M->data.fl[4]+=1.;
        warp_M->data.fl[6]=0;
        warp_M->data.fl[7]=0;
        warp_M->data.fl[8]=1.;

        // compose
        cvMatMul(warp_M, delta_M, comp_M);
        warp_p->data.fl[0] = comp_M->data.fl[0]-1.;
        warp_p->data.fl[1] = comp_M->data.fl[1];
        warp_p->data.fl[2] = comp_M->data.fl[2];
        warp_p->data.fl[3] = comp_M->data.fl[3];
        warp_p->data.fl[4] = comp_M->data.fl[4]-1.;
        warp_p->data.fl[5] = comp_M->data.fl[5];
      }
// fprintf(stderr, "warp_p: %f, %f\n",
//         warp_p->data.fl[0]*warp_p->data.fl[0]+
//         warp_p->data.fl[1]*warp_p->data.fl[1],
//         warp_p->data.fl[2]*warp_p->data.fl[2]+
//         warp_p->data.fl[3]*warp_p->data.fl[3]);
// cvPrintf(stderr, "%f ", warp_p);
    }
// CV_TIMER_SHOW();
// cvShowImageEx("Test", IWxp, CV_CM_GRAY); CV_WAIT();
cvCopy(warp_p, p_init);
  }
  
  void affine_fa(CvMat * img, CvMat * tmplt, CvMat * p_init,
                 const int maxiter, const double eps)
  {
    int nr=tmplt->rows,nc=tmplt->cols,iter;
    const int N_p=6;

    // initialize common parameters
    init_a(img, tmplt, p_init, N_p, nr, nc);

    if (!img_dx) { img_dx = cvCreateMat(img->rows, img->cols, CV_32F); }
    if (!img_dy) { img_dy = cvCreateMat(img->rows, img->cols, CV_32F); }

    // step 3: evaluate the gradient
    cvSobel(img, img_dx, 1, 0, 1); cvScale(img_dx, img_dx, 0.5);
    cvSobel(img, img_dy, 0, 1, 1); cvScale(img_dy, img_dy, 0.5);

    // step 4: evaluate the Jacobian dW_dp
    jacobian_a(dW_dp, nr, nc);

    double rms_error=0;

// CV_SHOW(tmplt);
// cvShowImageEx("Test", tmplt, CV_CM_GRAY); CV_WAIT();
// CV_TIMER_START();
    for (iter=0;iter<maxiter;iter++)
    {
      // step 1: warp I with the warp
      warp_a(img, IWxp, warp_p);
// CV_SHOW(IWxp);
// cvShowImageEx("Test", IWxp, CV_CM_GRAY); CV_WAIT();

      // step 2: compute error image
      // cvSub(tmplt, IWxp, error_img);
      rms_error=0;
      {
        float * tptr = tmplt->data.fl;
        float * wptr = IWxp->data.fl;
        float * eptr = error_img->data.fl;
        int tstep=tmplt->step/sizeof(float);
        int wstep=IWxp->step/sizeof(float);
        int estep=error_img->step/sizeof(float);
        int i,j;
        for (i=0;i<nr;i++){
          for (j=0;j<nc;j++){
            eptr[j]=tptr[j]-wptr[j];
            rms_error += eptr[j]*eptr[j];
          }
          tptr+=tstep; wptr+=wstep; eptr+=estep;
        }
        rms_error = sqrt(rms_error);
      }
// CV_SHOW(error_img);
// fprintf(stderr, "rms_error: %f\n\n", rms_error);
// cvShowImageEx("Test", error_img, CV_CM_GRAY); CV_WAIT();
      if (rms_error<eps) {break;}

      // step 3: warp nabla_I with the warp
      warp_a(img_dx, nabla_Ix, warp_p);
      warp_a(img_dy, nabla_Iy, warp_p);
// CV_SHOW(nabla_Ix);
// CV_SHOW(nabla_Iy);

      // step 4: evaluate Jacobian

      // step 5: compute steepest descent image - VI_dW_dp
      sd_images(dW_dp, nabla_Ix, nabla_Iy, VI_dW_dp, N_p);
// CV_SHOW(VI_dW_dp);

      // step 6: compute Hessian and inverse
      hessian(VI_dW_dp, H);
      if (cvInvert(H, H_inv, CV_LU)==0.0) {cvInvert(H, H_inv, CV_SVD);}
// cvPrintf(stderr, "%g ", H);
// cvPrintf(stderr, "%g ", H_inv);
// cvShowImageEx("Test", H, CV_CM_GRAY); CV_WAIT();
// cvShowImageEx("Test", H_inv, CV_CM_GRAY); CV_WAIT();
      
      // step 7: update steepest descent image
      sd_update(VI_dW_dp, error_img, sd_delta_p);
// cvShowImageEx("Test", sd_delta_p, CV_CM_GRAY); CV_WAIT();
// fprintf(stderr, "sd_delta_p:\n");
// cvPrintf(stderr, "%g ", sd_delta_p);
// cvPrintf(stderr, "%g ", H_inv);

      // step 8
      cvMatMul(H_inv, sd_delta_p, delta_p);
      {
        double * dpptr = delta_p->data.db;
        double * sdptr = sd_delta_p->data.db;
        cvZero(delta_p);
        {
          double * hptr=H_inv->data.db;
          int i,j;
          for (i=0;i<N_p;i++,hptr+=H_inv->cols) {
            for (j=0;j<N_p;j++) {
              dpptr[i] += hptr[j]*sdptr[j];
            }
          }
        }
      }
// fprintf(stderr, "delta_p:\n");
// cvPrintf(stderr, "%f ", delta_p);

      // step 9: update warp parameters
      // cvAdd(warp_p, &delta_p, warp_p);
      // for (int i=0;i<N_p;i++){
      //   warp_p->data.fl[i] += delta_p->data.fl[i];
      // }
      warp_p->data.fl[0] += delta_p->data.db[0];
      warp_p->data.fl[1] += delta_p->data.db[2];
      warp_p->data.fl[2] += delta_p->data.db[4];
      warp_p->data.fl[3] += delta_p->data.db[1];
      warp_p->data.fl[4] += delta_p->data.db[3];
      warp_p->data.fl[5] += delta_p->data.db[5];
// fprintf(stderr, "warp_p: %f, %f\n",
//         warp_p->data.fl[0]*warp_p->data.fl[0]+
//         warp_p->data.fl[1]*warp_p->data.fl[1],
//         warp_p->data.fl[2]*warp_p->data.fl[2]+
//         warp_p->data.fl[3]*warp_p->data.fl[3]);
// cvPrintf(stderr, "%f ", warp_p);
    }
// CV_TIMER_SHOW();
// cvShowImageEx("Test", IWxp, CV_CM_GRAY); CV_WAIT();
cvCopy(warp_p, p_init);
  }

  static void jacobian_a(CvMat * dW_dp, const int nr, const int nc)
  {
    assert(dW_dp->rows==nr*2);
    assert(dW_dp->cols==nc*6);
    int i,j;
    {
      cvZero(dW_dp);
      int step=dW_dp->cols;
      float * dxptr00 = dW_dp->data.fl;
      float * dxptr11 = dW_dp->data.fl + step*nr + nc;
      float * dyptr02 = dW_dp->data.fl +           nc*2;
      float * dyptr13 = dW_dp->data.fl + step*nr + nc*3;
      float * dzptr04 = dW_dp->data.fl +           nc*4;
      float * dzptr15 = dW_dp->data.fl + step*nr + nc*5;
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
  }

  static void warp_a(CvMat * img, CvMat * IWxp,
                     CvMat * warp_p)
  {
    assert( (warp_p->rows==2) && (warp_p->cols==3) );
#if 0
    cvGetQuadrangleSubPix(img, IWxp, warp_p);
#elif 0
    cvWarpAffine(img, IWxp, warp_p);
#else
    float * iptr = img->data.fl;
    float * wptr = IWxp->data.fl;
    float * pptr = warp_p->data.fl;
    int y,x,nr=IWxp->rows,nc=IWxp->cols,
        istep=img->step/sizeof(float),
        wstep=IWxp->step/sizeof(float);
    int xpos, ypos;
    float cp0=pptr[0], sp1=pptr[1], sp3=pptr[3], cp4=pptr[4];
    int ww=img->width-1, hh=img->height-1;
    for (y=0;y<nr;y++,wptr+=wstep){
      for (x=0;x<nc;x++){
        xpos = x*cp0+y*sp1+pptr[2];
        ypos = x*sp3+y*cp4+pptr[5];
        if ( (xpos>ww) || (xpos<0) || (ypos>hh) || (ypos<0) )
        {
          wptr[x]=0;
        }else{
          wptr[x] = (iptr+istep*ypos)[xpos];
        }
      }
    }
#endif
  }

  static void sd_images(CvMat * dW_dp,
                        CvMat * nabla_Ix, CvMat * nabla_Iy,
                        CvMat * VI_dW_dp, const int N_p)
  {
    int p,i,j,nr=nabla_Ix->rows,nc=nabla_Ix->cols;
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

  static void hessian(CvMat * VI_dW_dp, CvMat * H)
  {
#if 1
    cvZero(H);
    int i,j,m,n,N_p=H->rows,
        dnr=VI_dW_dp->rows,dnc=VI_dW_dp->cols,
        nc=VI_dW_dp->cols/H->cols;
    double * hptr = H->data.db;
    float * h1ptr, * h2ptr;
    int hstep = H->step/sizeof(double);
    int dstep = VI_dW_dp->step/sizeof(float);
    assert( H->cols==N_p );
    assert( (hstep==6) && (N_p==6) );
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
#else
    int i,j,N_p=H->rows,nr=VI_dW_dp->rows,nc=VI_dW_dp->cols/H->cols;
    CvMat * tmp=cvCreateMat(nr,nc,CV_32F);
    for (i=0;i<N_p;i++){
      for (j=0;j<N_p;j++){
        CvMat * mat1,mat1_stub, * mat2,mat2_stub;
        mat1=cvGetSubRect(VI_dW_dp, &mat1_stub,
                          cvRect(nc*i, 0, nc, nr));
        mat2=cvGetSubRect(VI_dW_dp, &mat2_stub,
                          cvRect(nc*j, 0, nc, nr));
        cvMul(mat1, mat2, tmp);
        CV_MAT_ELEM(*H, double, i, j)=cvSum(tmp).val[0];
      }
    }
    cvReleaseMat(&tmp);
#endif
  }

  static void sd_update(CvMat * VI_dW_dp, CvMat * error_img,
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

};
