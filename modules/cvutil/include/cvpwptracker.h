/**
 * @file   cvpwptracker.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Feb  6 16:35:26 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_PWP_TRACKER_H__
#define __CV_PWP_TRACKER_H__

#include "cvtracker.h"
#include "cvlevelset.h"
#include "cvinvcomp.h"
#include "cvshapeprior.h"
#include "cvshapedesc.h"

//#define WITH_TZG 
#if !defined(ANDROID) 
#include "cvparticlefilter.h" // for box2rect functions
#if defined(WITH_TZG)
#include "LevelsetTracking.h"
#endif // WITH_TZG
#endif // ANDROID

//#include "lkTracker.h"

void icvDriftCorrection(CvMat * warp_p, CvMat * bw,
                        int nr, int nc, float border, float multiplier);
void icvFillRegion(CvMat * mask);

class CvPWPTracker
{
  CvShapePrior prior;
  CvFourierDescriptor fdesc;
  
 //protected:
 public:
  int m_initialized;
  CvGenericTracker & m_t;

public://tzg  
  // -------------------------------------------------------
  // copied from generic tracker
  // -------------------------------------------------------
  CvSize & m_imsize;
  CvMat silhImage_stub;
  CvMat * silhImage;
  CvMat * imgY;
  CvMat * imgU;
  CvMat * imgV;
  CvMat * imgYpatch, * imgUpatch, * imgVpatch;
  int & m_iSqrMultiplier;
  int & m_iLinMultiplier;

  // -------------------------------------------------------
  // current tracker status
  // -------------------------------------------------------
  CvBox2D m_innerbox;
  CvBox2D m_outerbox;
  CvRect m_innerroi;
  CvRect m_outerroi;

  // -------------------------------------------------------
  //  segmentation with level-sets method
  // -------------------------------------------------------
  CvMat * phi;
  CvMatND * reffhist;
  CvMatND * refbhist;
  CvMatND * currhist;

  CvMat * Heaviside;
  CvMat * dirac;
  CvMat * Mf;
  CvMat * Mb;
  CvMat * Pf;
  CvMat * Pb;
  CvMat * backproj;
  CvMat * dPphi_dphi;
  CvMat * del2;
  CvMat * dx; 
  CvMat * dy; 
  CvMat * mag; 
  CvMat * kappa; 
  
  // -------------------------------------------------------
  // registration using inverse compositional algorithm
  // -------------------------------------------------------
  int N_p;
  
  CvMat * img_dx;
  CvMat * img_dy;
  CvMat * dW_dp; // Jacobian matrix
  CvMat * IWxp;
  CvMat * warp_p;

  CvMat * error_img;
  CvMat * nabla_Ix;
  CvMat * nabla_Iy;
  CvMat * VI_dW_dp;
  CvMat * Hessian;
  CvMat * H_inv;
  CvMat * sd_delta_p;
  CvMat * delta_p;

  CvMat * delta_M;
  CvMat * delta_M_inv;
  CvMat * warp_M;
  CvMat * comp_M;

  void copy_to_local()
  {
    // if (m_t.m_silhImage){
    //   silhImage = cvGetMat(m_t.m_silhImage, &silhImage_stub);
    // }
    imgY = m_t.m_imgY;
    imgU = m_t.m_imgU;
    imgV = m_t.m_imgV;
  }

  void init_a(const int nr, const int nc)
  {
    if (!dW_dp)
    {
      if (!img_dx) { img_dx = cvCreateMat(nr, nc, CV_32F); }
      if (!img_dy) { img_dy = cvCreateMat(nr, nc, CV_32F); }
      if (!dW_dp) { dW_dp = cvCreateMat(nr*2,nc*N_p,CV_32F); }
      icvCalcJacobian(dW_dp, nr, nc);
      if (!IWxp) { IWxp = cvCreateMat(nr,nc,CV_32F); }
      if (!error_img) { error_img = cvCreateMat(nr, nc, CV_32F); }
      if (!nabla_Ix) { nabla_Ix = cvCreateMat(nr, nc, CV_32F); }
      if (!nabla_Iy) { nabla_Iy = cvCreateMat(nr, nc, CV_32F); }
      if (!VI_dW_dp) { VI_dW_dp = cvCreateMat(nr, nc*N_p, CV_32F); }
    }

    if ( (dW_dp->rows!=nr*2) || (dW_dp->cols!=nc*N_p) )
    {
      if (img_dx) { cvReleaseMat(&img_dx); img_dx=NULL; }
      if (img_dy) { cvReleaseMat(&img_dy); img_dy=NULL; }
      if (dW_dp) { cvReleaseMat(&dW_dp); dW_dp=NULL; }
      if (IWxp) { cvReleaseMat(&IWxp); IWxp=NULL; }
      if (error_img) { cvReleaseMat(&error_img); error_img=NULL; }
      if (nabla_Ix) { cvReleaseMat(&nabla_Ix); nabla_Ix=NULL; }
      if (nabla_Iy) { cvReleaseMat(&nabla_Iy); nabla_Iy=NULL; }
      if (VI_dW_dp) { cvReleaseMat(&VI_dW_dp); VI_dW_dp=NULL; }
      // reallocate matrices
      if (!img_dx) { img_dx = cvCreateMat(nr, nc, CV_32F); }
      if (!img_dy) { img_dy = cvCreateMat(nr, nc, CV_32F); }
      if (!dW_dp) { dW_dp = cvCreateMat(nr*2,nc*N_p,CV_32F); }
      icvCalcJacobian(dW_dp, nr, nc);
      if (!IWxp) { IWxp = cvCreateMat(nr,nc,CV_32F); }
      if (!error_img) { error_img = cvCreateMat(nr, nc, CV_32F); }
      if (!nabla_Ix) { nabla_Ix = cvCreateMat(nr, nc, CV_32F); }
      if (!nabla_Iy) { nabla_Iy = cvCreateMat(nr, nc, CV_32F); }
      if (!VI_dW_dp) { VI_dW_dp = cvCreateMat(nr, nc*N_p, CV_32F); }
    }
  }

  void segparaminit(int nr, int nc)
  {
    if ( (!Heaviside) )
    {
      if (!Heaviside) { Heaviside = cvCreateMat(nr,nc,CV_32F); }
      if (!dirac) { dirac = cvCreateMat(nr,nc,CV_32F); }
      if (!Mf)    { Mf = cvCreateMat(nr,nc,CV_32F); }
      if (!Mb)    { Mb = cvCreateMat(nr,nc,CV_32F); }
      if (!Pf)    { Pf = cvCreateMat(nr,nc,CV_32F); }
      if (!Pb)    { Pb = cvCreateMat(nr,nc,CV_32F); }
      if (!backproj) { backproj = cvCreateMat(nr, nc, CV_32F); }
      if (!dPphi_dphi) { dPphi_dphi = cvCreateMat(nr,nc,CV_32F); }
      if (!del2) { del2 = cvCreateMat(nr, nc, CV_32F); }
      if (!dx)   { dx = cvCreateMat(nr, nc, CV_32F); }
      if (!dy)   { dy = cvCreateMat(nr, nc, CV_32F); }
      if (!mag)  { mag = cvCreateMat(nr, nc, CV_32F); }
      if (!kappa){ kappa = cvCreateMat(nr, nc, CV_32F);}
    }

    // re-initialize size of phi
    if ( (nr!=Heaviside->rows) || (nc!=Heaviside->cols) )
    {
      cvReleaseMat(&Heaviside);                   Heaviside=NULL;
      cvReleaseMat(&dirac);                       dirac=NULL;
      cvReleaseMat(&Mf);                          Mf=NULL;
      cvReleaseMat(&Mb);                          Mb=NULL;
      cvReleaseMat(&Pf);                          Pf=NULL;
      cvReleaseMat(&Pb);                          Pb=NULL;
      cvReleaseMat(&backproj);                    backproj=NULL;
      cvReleaseMat(&dPphi_dphi);                  dPphi_dphi=NULL;
      cvReleaseMat(&del2);                        del2=NULL;
      cvReleaseMat(&dx);                          dx=NULL;
      cvReleaseMat(&dy);                          dy=NULL;
      cvReleaseMat(&mag);                         mag=NULL;
      cvReleaseMat(&kappa);                       kappa=NULL;
      if (!Heaviside) { Heaviside = cvCreateMat(nr,nc,CV_32F); }
      if (!dirac) { dirac = cvCreateMat(nr,nc,CV_32F); }
      if (!Mf)    { Mf = cvCreateMat(nr,nc,CV_32F); }
      if (!Mb)    { Mb = cvCreateMat(nr,nc,CV_32F); }
      if (!Pf)    { Pf = cvCreateMat(nr,nc,CV_32F); }
      if (!Pb)    { Pb = cvCreateMat(nr,nc,CV_32F); }
      if (!backproj) { backproj = cvCreateMat(nr, nc, CV_32F); }
      if (!dPphi_dphi) { dPphi_dphi = cvCreateMat(nr,nc,CV_32F); }
      if (!del2) { del2 = cvCreateMat(nr, nc, CV_32F); }
      if (!dx)   { dx = cvCreateMat(nr, nc, CV_32F); }
      if (!dy)   { dy = cvCreateMat(nr, nc, CV_32F); }
      if (!mag)  { mag = cvCreateMat(nr, nc, CV_32F); }
      if (!kappa){ kappa = cvCreateMat(nr, nc, CV_32F);}
    }
  }

  // int registration0();
  // int registration1();
  
 // protected:
  int segmentation();
  int registration();
  
 public:
  CvPWPTracker(CvGenericTracker & t):
      fdesc(22),
      m_initialized(0),
      m_t(t), // reference to generic tracker
      m_imsize(t.m_imsize),
      silhImage(NULL),
      imgY(NULL), imgU(NULL), imgV(NULL),
      imgYpatch(NULL), imgUpatch(NULL), imgVpatch(NULL),
      m_iSqrMultiplier(t.m_iSqrMultiplier),
      m_iLinMultiplier(t.m_iLinMultiplier),
      // common variables
      phi(NULL), bw(NULL), 
      reffhist(NULL), refbhist(NULL), currhist(NULL),
      // segmentation variables
      Heaviside(NULL), dirac(NULL),
      Mf(NULL),Mb(NULL),Pf(NULL),Pb(NULL),
      backproj(NULL),dPphi_dphi(NULL),
      del2(NULL),dx(NULL), dy(NULL), mag(NULL), kappa(NULL), // on phi
      // tracking variables
      N_p(4),img_dx(NULL),img_dy(NULL),
      dW_dp(NULL), // Jacobian matrix
      IWxp(NULL),warp_p(NULL),
      error_img(NULL),nabla_Ix(NULL),nabla_Iy(NULL),
      VI_dW_dp(NULL),
      Hessian(NULL),H_inv(NULL),
      sd_delta_p(NULL),delta_p(NULL),
      delta_M(NULL),delta_M_inv(NULL),
      warp_M(NULL),comp_M(NULL),
      m_status(2),
      bw_full(NULL),phi_full(NULL),Pf_full(NULL),Pb_full(NULL),
      invwarp_p3x3(NULL)
  {
    // histograms for segmentation
    int dims[]={32,32,32};
    reffhist = cvCreateMatND(3, (int*)dims, CV_32F); cvZero(reffhist);
    refbhist = cvCreateMatND(3, (int*)dims, CV_32F); cvZero(refbhist);
    currhist = cvCreateMatND(3, (int*)dims, CV_32F); cvZero(currhist);

    // variables with fixed sizes for registration
    if (!Hessian) { Hessian = cvCreateMat(N_p, N_p, CV_64F); } // double
    if (!warp_p) { warp_p = cvCreateMat(N_p, 1, CV_32F); }
    if (!H_inv) { H_inv = cvCreateMat(N_p, N_p, CV_64F); }  // double
    if (!sd_delta_p) { sd_delta_p = cvCreateMat(N_p, 1, CV_64F); } // double
    if (!delta_p) { delta_p = cvCreateMat(N_p, 1, CV_64F); } // double 
    if (!delta_M) { delta_M = cvCreateMat(3, 3, CV_32F); }
    if (!delta_M_inv) { delta_M_inv = cvCreateMat(3, 3, CV_32F); }
    if (!warp_M) { warp_M = cvCreateMat(3, 3, CV_32F); }
    if (!comp_M) { comp_M = cvCreateMat(3, 3, CV_32F); }
    if (!invwarp_p3x3) { invwarp_p3x3 = cvCreateMat(3,3,CV_32F); }

    // prior.load((char*)"../data/shapeprior-meanshape.raw",
    //            (char*)"../data/shapeprior-mean.raw",
    //            (char*)"../data/shapeprior-pc.raw",
    //            (char*)"../data/shapeprior-latent.raw");
  }

  ~CvPWPTracker()
  {
    // original frames
    if (imgYpatch) { cvReleaseMat(&imgYpatch); imgYpatch=NULL; }
    if (imgUpatch) { cvReleaseMat(&imgUpatch); imgUpatch=NULL; }
    if (imgVpatch) { cvReleaseMat(&imgVpatch); imgVpatch=NULL; }
    
    // common variables
    if (phi){ cvReleaseMat(&phi); phi=NULL; }
    if (phi_full){ cvReleaseMat(&phi_full); phi_full=NULL; }
    if (bw){ cvReleaseMat(&bw); bw=NULL; }
    if (reffhist) { cvReleaseMatND(&reffhist); reffhist=NULL; }
    if (refbhist) { cvReleaseMatND(&refbhist); refbhist=NULL; }
    if (currhist) { cvReleaseMatND(&currhist); currhist=NULL; }

    // release variables for tracking with inverse compositional algo.
    if (Heaviside) { cvReleaseMat(&Heaviside);  Heaviside=NULL; }
    if (dirac) { cvReleaseMat(&dirac); dirac=NULL; }
    if (Mf) { cvReleaseMat(&Mf); Mf=NULL; }
    if (Mb) { cvReleaseMat(&Mb); Mb=NULL; }
    if (Pf) { cvReleaseMat(&Pf); Pf=NULL; }
    if (Pb) { cvReleaseMat(&Pb); Pb=NULL; }
    if (backproj) { cvReleaseMat(&backproj); backproj=NULL; }
    if (dPphi_dphi) { cvReleaseMat(&dPphi_dphi); dPphi_dphi=NULL; }
    if (del2) { cvReleaseMat(&del2); del2=NULL; }
    if (dx) { cvReleaseMat(&dx); dx=NULL; }
    if (dy) { cvReleaseMat(&dy); dy=NULL; }
    if (mag) { cvReleaseMat(&mag); mag=NULL; }
    if (kappa) { cvReleaseMat(&kappa); kappa=NULL; }

    // release variables for segmentation with level-set method
    if (img_dx) { cvReleaseMat(&img_dx); img_dx=NULL; }
    if (img_dy) { cvReleaseMat(&img_dy); img_dy=NULL; }
    if (dW_dp) { cvReleaseMat(&dW_dp); dW_dp=NULL; }
    if (IWxp) { cvReleaseMat(&IWxp); IWxp=NULL; }
    if (error_img) { cvReleaseMat(&error_img); error_img=NULL; }
    if (nabla_Ix) { cvReleaseMat(&nabla_Ix); nabla_Ix=NULL; }
    if (nabla_Iy) { cvReleaseMat(&nabla_Iy); nabla_Iy=NULL; }
    if (VI_dW_dp) { cvReleaseMat(&VI_dW_dp); VI_dW_dp=NULL; }
    if (Hessian) { cvReleaseMat(&Hessian); Hessian=NULL; }
    if (H_inv) { cvReleaseMat(&H_inv); H_inv=NULL; }
    if (sd_delta_p) { cvReleaseMat(&sd_delta_p); sd_delta_p=NULL; }
    if (delta_p) { cvReleaseMat(&delta_p); delta_p=NULL; }
    if (delta_M) { cvReleaseMat(&delta_M); delta_M=NULL; }
    if (delta_M_inv) { cvReleaseMat(&delta_M_inv); delta_M_inv=NULL; }
    if (warp_M) { cvReleaseMat(&warp_M); warp_M=NULL; }
    if (warp_p) { cvReleaseMat(&warp_p); warp_p=NULL; }
    if (comp_M) { cvReleaseMat(&comp_M); comp_M=NULL; }

    if (bw_full) { cvReleaseMat(&bw_full); bw_full=NULL; }
    if (Pf_full) { cvReleaseMat(&Pf_full); Pf_full=NULL; }
    if (Pb_full) { cvReleaseMat(&Pb_full); Pb_full=NULL; }
    if (phi_full) { cvReleaseMat(&phi_full); phi_full=NULL; }
    if (invwarp_p3x3) { cvReleaseMat(&invwarp_p3x3); invwarp_p3x3=NULL; }
  }

  /** 
   * evaluate the given location and start tracking ..
   * 
   * @param roi     in:  detected location, for initialize tracking
   * 
   * @return status code for detection result
   */
  int initialize(CvRect roi);
  inline int initialized() {return m_initialized;}

#if defined(WITH_TZG)
  int myInitialize(CvRect roi);
  int myRegistration();
  int mySegmentation();
  int myRecognition();
  int myDisplay(IplImage* imgTest);
  void checkTrackFalse();
  void addSkinOffset(IplImage* imgRGB, IplImage* imgSil);
  int m_updateFrameNum;
  CLevelsetTracking myLevelsetTk;
#endif // ANDROID

  //CLKTracker m_lkTrack;

  /** 
   * update the tracking status, assuming tracker initialized
   * 
   * @return status code
   */
  int update()
  {
    copy_to_local();
	return 1;
  }

  CvBox2D innerbox() {return m_innerbox;} // !!! fix me here !!!
  CvBox2D outerbox() {return m_outerbox;} // !! update outerbox/outerroi 
  CvRect innerrect() {return m_innerroi;} // !! by function 
  CvRect outerrect() {return m_outerroi;}

  int m_status;
  // int status() {return m_status;}
  int status() {return prior.status();}
  
  CvMat * bw;
  CvMat * bw_full;
  CvMat * phi_full;
  CvMat * Pf_full;
  CvMat * Pb_full;
  CvMat * invwarp_p3x3;
};

#endif // __CV_PWP_TRACKER_H__



/*
int CvPWPTracker::registration()
{
  const int invcomp_flag = 0;

  if (invcomp_flag) {
    return registration0();
  }else{
    return registration1();
  }
}

int CvPWPTracker::registration0()
{
  int nr=phi->rows, nc=phi->cols;
  const int border = 4*m_iLinMultiplier;
  int hh=nr-border*2, ww=nc-border*2;
  int iter;
  const int maxiter = 5;
  // CvMat * tmplt = cvCreateMat(nr, nc, CV_32F);
  CvMat * tmplt = cvCreateMat(hh, ww, CV_32F);

  //---------------------------------
  // warp larger templates into full frame 
  //---------------------------------
  {
    if (!bw_full) {
      bw_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_8U);
    }
    // if (!phi_full) {
    //   phi_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_32F);
    // }
    // assert( (warp_p->cols==3)&&(warp_p->rows==2) );
    
    CvMat * invwarp_p3x3 = cvCreateMat(3,3,CV_32F);
    icvWarpInvert(warp_p, invwarp_p3x3);
    icvWarp(bw, bw_full, invwarp_p3x3);
    // icvWarp(phi, phi_full, invwarp_p3x3);
    cvReleaseMat(&invwarp_p3x3);
  }

  // initialize variable matrices with template size
  init_a(hh, ww); 

  // get template from previous frame
  // icvWarp(currImage, tmplt, warp_p);
  icvWarp(currImage, tmplt, warp_p);
  // cvShowImageEx("Test", tmplt, CV_CM_GRAY); CV_WAIT();

  //---------------------------------
  // step 3: evaluate the gradient of the template
  //---------------------------------
  cvSobel(tmplt, img_dx, 1, 0, 1); cvScale(img_dx, img_dx, 0.5);
  cvSobel(tmplt, img_dy, 0, 1, 1); cvScale(img_dy, img_dy, 0.5);

  //---------------------------------
  // step 4: evaluate the Jacobian dW_dp
  //---------------------------------
  // icvCalcJacobian(dW_dp, nr, nc);
  icvCalcJacobian(dW_dp, hh, ww);

  //---------------------------------
  // step 5: compute the steepest descent images
  //---------------------------------
  icvCalcStDescImages(dW_dp, img_dx, img_dy, VI_dW_dp);
  // cvShowImageEx("Test", dW_dp, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", VI_dW_dp, CV_CM_GRAY); CV_WAIT();

  {
    //---------------------------------
    // step 6: compute the Hessian matrix
    //---------------------------------
    icvCalcHessian(VI_dW_dp, Hessian);

    if (cvInvert(Hessian, H_inv, CV_LU)==0.0) {
      cvInvert(Hessian, H_inv, CV_SVD);
    }
    // cvShowImageEx("Test", H_inv, CV_CM_GRAY); CV_WAIT();
  }
  
  double rms_error=0;

  // cvShowImageEx("Test", tmplt, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", currImage, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", H_inv, CV_CM_GRAY); CV_WAIT();
  for (iter=0;iter<maxiter;iter++)
  {
    // step 2: compute the error image I(W(x;p))-T(x)
    rms_error=0;

    // -------------------------------------------------------
    // inverse compositional -- tracking section
    // -------------------------------------------------------
    {
      // step 1: warp I with W(x;p) to compute I(W(x;p))
      icvWarp(nextImage, IWxp, warp_p);
      
      float * tptr = tmplt->data.fl;
      float * wptr = IWxp->data.fl;
      float * eptr = error_img->data.fl;
      // int tstep=tmplt->step/sizeof(float);
      // int wstep=IWxp->step/sizeof(float);
      // int estep=error_img->step/sizeof(float);
      int step = error_img->step/sizeof(float);
      // assert(step==Pf->step/sizeof(float));
      // assert(step==Pb->step/sizeof(float));
      assert(step==tmplt->step/sizeof(float));
      assert(step==IWxp->step/sizeof(float));
      int i,j;
      for (i=0;i<hh;i++)
      {
        for (j=0;j<ww;j++)
        {
          eptr[j]=tptr[j]-wptr[j];
          rms_error += eptr[j]*eptr[j];
        }
        tptr+=step; wptr+=step;
        eptr+=step;
      }
      rms_error = sqrt(rms_error);

      // step 7: update steepest descent images
      icvUpdateStDescImages(VI_dW_dp, error_img, sd_delta_p);

      // step 8: compute delta_p
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
    }

    // termination criteria
    {
      double dpsum=0;
      double * dpptr = delta_p->data.db;
      for (int i = 0; i < N_p; i++) {dpsum+=dpptr[i]*dpptr[i];}
      // fprintf(stderr, "delta_p: %f\n", dpsum);
      // cvPrintf(stderr, "%f, " , delta_p);
      // if ( sqrt(dpsum) < 0.08 ) {
      if ( sqrt(dpsum) < 0.2 ) {
        fprintf(stderr, "delta_p[%d]: %f\n", iter, dpsum);
        break;
      }
    }

    // step 9: update the warp 
    {
      icvWarpTranspose(delta_p, delta_M);

      // invert compositional warp
      if (cvInvert(delta_M, delta_M_inv, CV_LU)==0.0) {
        cvInvert(delta_M, delta_M_inv, CV_SVD);
      }

      // current warp
      icvWarpReshape(warp_p, warp_M);

      // compose
      cvMatMul(warp_M, delta_M, comp_M);
      icvWarpCompose(comp_M, warp_p);
    }

  } // end of iteration
  
  // ---------------------------------
  // drift correction
  icvDriftCorrection(warp_p, bw, nr,nc,hh,ww,border,m_iLinMultiplier);

  // ---------------------------------
  // new warped phi to reduce segmentation iterations
  // icvWarp(phi_full, phi, warp_p);

  // ---------------------------------
  // convert warp_p to box representation
  if (1)
  {
    CvPoint2D32f pts[4]={0,};
    double angle = icvWarpToPoints(warp_p, pts, nr, nc);

    m_outerbox.angle = angle;
    m_outerbox.center.x = (pts[0].x+pts[2].x)*0.5;
    m_outerbox.center.y = (pts[0].y+pts[2].y)*0.5;
    m_outerbox.size.width  = pts[1].x-pts[0].x;
    m_outerbox.size.height = pts[3].y-pts[0].y;
    m_innerbox = m_outerbox;
    m_innerbox.size.width  -= border*2.;
    m_innerbox.size.height -= border*2.;
  }

  if (tmplt) { cvReleaseMat(&tmplt); }

  return 1;
}
*/
