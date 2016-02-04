/**
 * @file   cvpwptracker.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Feb  6 16:35:06 2013
 * 
 * @brief  implementation of the Pixel-Wise Posterior (PWP)
 *         based tracking system
 */

#include "cvpwptracker.h"

#define CV_PWP_USE_OPENMP         0 
#define CV_PWP_USE_NARROWBAND     0

// #if CV_PWP_USE_OPENMP
// #include <omp.h>
// #endif

// extern void printminmax(CvMat *);
void icvCalcHistFromMaskedRegion(
    CvMat * imgY, CvMat * imgU, CvMat * imgV,
    CvMatND * fghist, CvMatND * bghist, CvMat * submask);

void icvCalcWarpPixelWisePosterior(
    CvMat * imgY, CvMat * imgU, CvMat * imgV,
    CvMatND * reffhist, CvMatND * refbhist, double etaf, double etab,
    CvMat * Pf_full, CvMat * Pb_full, CvMat * Pf, CvMat * Pb,
    CvMat * invwarp_p3x3);

int CvPWPTracker::initialize(CvRect roi)
{
  m_initialized=0;
  copy_to_local();

  static const float bbb=6.*m_iLinMultiplier;
  m_outerroi = roi;
  m_outerroi.x = roi.x-bbb;
  m_outerroi.y = roi.y-bbb;
  m_outerroi.width  = roi.width +bbb*2.;
  m_outerroi.height = roi.height+bbb*2.;
  if ( (m_outerroi.x<0) || (m_outerroi.y<0) ||
       ((m_outerroi.width+m_outerroi.x)>m_imsize.width) ||
       ((m_outerroi.height+m_outerroi.y)>m_imsize.height) )
  {
    fprintf(stderr, "WARNING: fail to initialize - on the boundary\n");
    return 0;
  } // fail to initialize - on the boundary

  m_outerbox = cvBox2DFromRect(m_outerroi);

  {
    CvPoint2D32f pts[4];
    cvBoxPoints32f(m_outerbox, pts);
    icvWarpInit(warp_p,pts[0].x,pts[0].y);
  }

  CvBox2D box = m_outerbox;

  // ---------------------------------
  // calculate histogram of current frame
  {
    CvMat imgY_stub,imgU_stub,imgV_stub;
    // box.size.width =MIN(24*m_iLinMultiplier,box.size.width);
    // box.size.height=MIN(30*m_iLinMultiplier,box.size.height);
    box.size.width =MIN(30*m_iLinMultiplier, box.size.width);
    box.size.height=MIN(36*m_iLinMultiplier, box.size.height);
    CvRect roiex = m_outerroi;

    // assert( phi==NULL && bw==NULL);
    if (phi || bw)
    {
      if (phi) {cvReleaseMat(&phi); phi=NULL;}
      if (bw)  {cvReleaseMat(&bw ); bw =NULL;}
      if (!phi) { phi = cvCreateMat(roiex.height, roiex.width, CV_32F); }
      if (!bw)  { bw  = cvCreateMat(roiex.height, roiex.width, CV_8U ); }
    }else{
      if (!phi) { phi = cvCreateMat(roiex.height, roiex.width, CV_32F); }
      if (!bw)  { bw  = cvCreateMat(roiex.height, roiex.width, CV_8U ); }
    }

    //if ((!imgYpatch)||(roiex.height!=imgYpatch->height))
    { 
imgYpatch = cvCloneMat(cvGetSubRect(imgY, &imgY_stub, roiex)); 
imgUpatch = cvCloneMat(cvGetSubRect(imgU, &imgU_stub, roiex)); 
imgVpatch = cvCloneMat(cvGetSubRect(imgV, &imgV_stub, roiex)); 
	}

    CvMat submask_stub,*submask;
    //submask   = cvCloneMat(cvGetSubRect(silhImage, &submask_stub, roiex));
    submask = cvCreateMat(roiex.height,roiex.width,CV_8U);
    cvZero(submask);
    {
      CvMat * subsubmask=
          cvGetSubRect(submask,&submask_stub,
                       cvRect(bbb,bbb,roi.width,roi.height));
      cvSet(subsubmask,cvScalar(255));
    }

// cvShowImage("Test", imgYpatch); CV_WAIT();
// cvShowImage("Test", submask);   CV_WAIT();
    // icvFillRegion(submask);

// cvShowImage("Test", submask); CV_WAIT();
    // calculate histogram of current frame
	assert( (submask->cols==imgYpatch->cols) &&
            (submask->rows==imgYpatch->rows) );
    icvCalcHistFromMaskedRegion(
        imgYpatch, imgUpatch, imgVpatch, reffhist, refbhist, submask);
    // initialize phi !~
    icvInitializeLevelSet(submask, phi);

    // release memory
    cvReleaseMat(&imgYpatch); imgYpatch=NULL;
    cvReleaseMat(&imgUpatch); imgUpatch=NULL;
    cvReleaseMat(&imgVpatch); imgVpatch=NULL;

    // ---------------------------------
    // additional segmentation steps
    // for accurate initial pixel-wise posterior
    for (int i=0;i<10;i++)
    {
    // initialize segmentation !~
    segmentation();
    //cvCmpS(Heaviside,0.5,submask,CV_CMP_GT);
    cvCmpS(Heaviside,0.7,submask,CV_CMP_GT);
    // calculate histogram of current frame
    icvCalcHistFromMaskedRegion(
        imgYpatch, imgUpatch, imgVpatch, reffhist, refbhist, submask);
    // initialize phi !~
    icvInitializeLevelSet(submask, phi);
    }
// cvShowImage("Test", submask); CV_WAIT();

    // release memory
    cvReleaseMat(&imgYpatch); imgYpatch=NULL;
    cvReleaseMat(&imgUpatch); imgUpatch=NULL;
    cvReleaseMat(&imgVpatch); imgVpatch=NULL;
    cvReleaseMat(&submask);
  }
  
  m_initialized=1;
  return 1;
}

//-------------------------------------------------------
// SEGMENTATION
//-------------------------------------------------------
int CvPWPTracker::segmentation()
{
  CvRect roi = m_outerroi;
  if (phi==0){ assert(phi); return 0; }
  const int maxiter = 15, maxiter2 = 10;//initialized()?10:15;
  int nr = phi->rows,nc=phi->cols;

  segparaminit(nr,nc);
  
  assert( (nr==roi.height) && (nc==roi.width) );

  // if (!initialized())
  {
    if (!imgYpatch) { imgYpatch = cvCreateMat(nr, nc, CV_8U); }
    if (!imgUpatch) { imgUpatch = cvCreateMat(nr, nc, CV_8U); }
    if (!imgVpatch) { imgVpatch = cvCreateMat(nr, nc, CV_8U); }
    icvWarp(imgY, imgYpatch, warp_p);
    icvWarp(imgU, imgUpatch, warp_p);
    icvWarp(imgV, imgVpatch, warp_p);

    // ---------------------------------
    // calculate histogram of current frame
    {
      cvZero(currhist);
      uchar * yptr=imgYpatch->data.ptr;
      uchar * uptr=imgUpatch->data.ptr;
      uchar * vptr=imgVpatch->data.ptr;
      int ystep=currhist->dim[0].step/sizeof(float),
          ustep=currhist->dim[1].step/sizeof(float),
          vstep=currhist->dim[2].step/sizeof(float);
      float * currptr=currhist->data.fl;
      int j,k,step=imgYpatch->step/sizeof(uchar);
      assert((ystep==1024)&&(ustep==32));
      for (j=0;j<nr;j++)
      {
// #if CV_PWP_USE_OPENMP
// #pragma omp parallel for shared(currptr,yptr,uptr,vptr) private(k)
// #endif
        for (k=0;k<nc;k++)
        {
          (currptr+
           ((int(yptr[k])>>3)<<10)+
           ((int(uptr[k])>>3)<<5))[int(vptr[k])>>3] += 1;
        }
        yptr+=step;uptr+=step;vptr+=step;
      }
      // normalize histogram 
      cvScale(currhist, currhist, 1./cvSum(currhist).val[0]);
    }

    // ---------------------------------
    // calculate pixel-wise posterior 
    {
      cvZero(Mf);cvZero(Mb);
      float * Mfptr=Mf->data.fl;
      float * Mbptr=Mb->data.fl;
      uchar * yptr=imgYpatch->data.ptr;
      uchar * uptr=imgUpatch->data.ptr;
      uchar * vptr=imgVpatch->data.ptr;
      int ystep=reffhist->dim[0].step/sizeof(float),
          ustep=reffhist->dim[1].step/sizeof(float),
          vstep=reffhist->dim[2].step/sizeof(float);
      float * reffptr=reffhist->data.fl;
      float * refbptr=refbhist->data.fl;
      int j,k,step=imgYpatch->step/sizeof(uchar);
      float reffval,refbval;
      int ypos, upos, vpos, yupos;
      double invrefsum;

      for (j=0;j<nr;j++)
      {
// #if CV_PWP_USE_OPENMP
// #pragma omp parallel for shared(yptr,uptr,vptr,ystep,ustep) \
//   private(ypos,upos,vpos,yupos,k,reffval,refbval,invrefsum)
// #endif 
        for (k=0;k<nc;k++)
        {
          ypos = int(yptr[k])>>3;
          upos = int(uptr[k])>>3;
          vpos = int(vptr[k])>>3;
          yupos = ystep*ypos+ustep*upos;
          reffval=(reffptr+yupos)[vpos];
          refbval=(refbptr+yupos)[vpos];

          invrefsum = 1./(reffval+refbval+1e-5);
          Mfptr[k]=MAX(MIN(reffval*invrefsum, 1.f),1E-5f);
          Mbptr[k]=MAX(MIN(refbval*invrefsum, 1.f),1E-5f);
        }
        yptr+=step;uptr+=step;vptr+=step;Mfptr+=step;Mbptr+=step;
      }
    }
  }
  
  int iter,iter2;
  int prevsum=-1,currsum;
  // cvSmooth(phi, phi, CV_GAUSSIAN, 3, 3);
  CvMat * prterm1 = phi;//prior.shapeterm1(phi);

  {
    CvMat * prterm1bw = cvCreateMat(phi->rows,phi->cols,CV_8U);
    cvCmpS(prterm1,0.1,prterm1bw,CV_CMP_GT);
    prterm1=prior.classify(prterm1bw);
    //fprintf(stderr, "%f\n", mystatus);
    //cvShowImage("bw",prterm1bw); 
    cvReleaseMat(&prterm1bw);
  }

  for (iter2=0; iter2<maxiter2; iter2++)
  {
  for (iter=0; iter<maxiter; iter++)
  {
    currsum=0;
    cvNeumannBoundCond(phi);
    
    // cvCalcHeaviside(phi, Heaviside, nc*0.06);
    // cvCalcDirac(phi, dirac, nc*0.06);
    cvCalcHeaviside(phi, Heaviside, 3.);
    cvCalcDirac(phi, dirac, 3.);

    double etaf = cvSum(Heaviside).val[0];
    double etab = nc*nr-etaf;

    // ---------------------------------
    // compute {Pf, Pb}, requires {Mf, Mb, etaf, etab}
    {
      float * Pfptr=Pf->data.fl;
      float * Pbptr=Pb->data.fl;
      float * Mfptr=Mf->data.fl;
      float * Mbptr=Mb->data.fl;
      int j,k,step=Pf->step/sizeof(float);
      double tmp,tf,tb,invtftb;
      for (j=0;j<nr;j++)
      {
        for (k=0;k<nc;k++)
        {
          tmp=1./(etaf*Mfptr[k]+etab*Mbptr[k]);
          tf=Mfptr[k]*tmp; tb=Mbptr[k]*tmp;
          Pfptr[k]=MAX(tf, 1E-5);
          Pbptr[k]=MAX(tb, 1E-5);
        }
        Pfptr+=step;Pbptr+=step;Mfptr+=step;Mbptr+=step;
      }
    }

    // ---------------------------------
    // compute { dPphi_dphi }, requires {Pf, Pb, dirac, Heaviside}
    {
      cvZero(dPphi_dphi);
      float * dPphiptr = dPphi_dphi->data.fl;
      float * diracptr = dirac->data.fl;
      float * Hptr = Heaviside->data.fl;
      float * Pfptr = Pf->data.fl;
      float * Pbptr = Pb->data.fl;
      int j,k,step = dPphi_dphi->step/sizeof(float);
      double pxval;
      for (j=0;j<nr;j++)
      {
        for (k=0;k<nc;k++)
#if CV_PWP_USE_NARROWBAND
          if (diracptr[k]!=0.0)
#endif // CV_PWP_USE_NARROWBAND
          {
            pxval=Hptr[k]*Pfptr[k]+(1.-Hptr[k])*Pbptr[k];
            dPphiptr[k]=diracptr[k]*(Pfptr[k]-Pbptr[k])/pxval;
          }
        dPphiptr+=step;//pxptr+=step;
        diracptr+=step;Hptr+=step;Pfptr+=step;Pbptr+=step;
      }
    }

    {
      // ---------------------------------
      // calculate gradient and laplacian of phi
      {
        cvZero(dx);cvZero(dy);cvZero(del2);cvSet(mag,cvScalar(1e-5));
        float * dxptr=dx->data.fl;
        float * dyptr=dy->data.fl;
        float * del2ptr=del2->data.fl;
        float * phiptr=phi->data.fl;
        float * magptr=mag->data.fl;
        float * diracptr=dirac->data.fl;
        int j,k,step=nc;
        assert((dirac->step/sizeof(float))==step);
        for (j=1;j<nr-1;j++)
        {
          dxptr+=step;dyptr+=step;phiptr+=step;del2ptr+=step;magptr+=step;
          diracptr+=step;
          for (k=1;k<nc-1;k++)
#if CV_PWP_USE_NARROWBAND
            if (diracptr[k]!=0)
#endif // CV_PWP_USE_NARROWBAND
          {
            dxptr[k]=(phiptr[k+1]-phiptr[k-1])*0.5;
            dyptr[k]=((phiptr+step)[k]-(phiptr-step)[k])*0.5;
            del2ptr[k]= // laplacian operator
                -(phiptr[k+1]+phiptr[k-1]
                  +(phiptr+step)[k]+(phiptr-step)[k]
                  -phiptr[k]*4.);
            magptr[k]=
                sqrt(dxptr[k]*dxptr[k]+dyptr[k]*dyptr[k])+FLT_EPSILON;
            // assert(!issanenum(magptr[k]));
            // assert(!issanenum(dyptr[k]));
            // if (phiptr[k]!=0)
            //   phiptr[k] +=
            //       (phiptr[k]>0?-1.0:1.0)*(1.0-magptr[k]+FLT_EPSILON);
          }
        }
      }

      cvDiv(dx,mag,dx);
      cvDiv(dy,mag,dy);

      // ---------------------------------
      // calc divergence
      // substract del2 with kappa
      {
        cvZero(kappa);
        float * kptr=kappa->data.fl;
        float * dxptr=dx->data.fl;
        float * dyptr=dy->data.fl;
        int j,k,step=nc;
        for (j=1;j<nr-1;j++)
        {
          kptr+=step;dxptr+=step;dyptr+=step;
          for (k=1;k<nc-1;k++)
          {
            // assert(!issanenum(dyptr[k]));
            kptr[k]=
                (dxptr[k+1]-dxptr[k-1]+(dyptr+step)[k]-(dyptr-step)[k])*0.5;
            // assert(!issanenum(kptr[k]));
          }
        }
      }
    }

    // ---------------------------------
    // add up the data term and smoothness term
    {
      float * pt1ptr = prterm1->data.fl;
      float * phiptr = phi->data.fl;
      float * diracptr=dirac->data.fl;
      float * kappaptr=kappa->data.fl;
      float * del2ptr= del2->data.fl;
      float * dphiptr= dPphi_dphi->data.fl;
      int j,k,step=phi->step/sizeof(float);
      for (j=0;j<nr;j++)
      {
// #if CV_PWP_USE_OPENMP
// #pragma omp parallel for private(k)                                \
//   shared(phiptr,dphiptr,diracptr,pt1ptr,del2ptr,kappaptr,currsum)  
// #endif // CV_PWP_USE_OPENMP
        for (k=0;k<nc;k++)
        {
#if CV_PWP_USE_NARROWBAND
          if (diracptr[k]!=0.0)
#endif // CV_PWP_USE_NARROWBAND
          {
            phiptr[k]+=(+dphiptr[k]                       // data term
                        +0.1*diracptr[k]*(pt1ptr[k]-phiptr[k]) // shape term
                        -0.02*(del2ptr[k]-kappaptr[k])    // distance term
                        +0.8*diracptr[k]*kappaptr[k]      // edge smooth
                        )*1.;                             // time step
          }
          // assert(!issanenum(phiptr[k]));
          if (phiptr[k]>0) {currsum++;}
        }
        phiptr+=step;del2ptr+=step;dphiptr+=step;
        kappaptr+=step;diracptr+=step;
        pt1ptr+=step;
      }
    }

    // ---------------------------------
    // criteria for termination of the iteration
    {
      if (currsum==prevsum){
        break;
      }else{
        prevsum=currsum;
      }
    }

    // show binary image
    if (0)
    {
      // cvShowImageEx("Test", phi); CV_WAIT();
      cvCmpS(phi, 0, bw, CV_CMP_GT);
      //cvShowImage("Test", bw); CV_WAIT();
    }
  }
  if (iter==0){break;}
  }
  //fprintf(stderr, "segmentation[%d,%d]\n", iter2,iter);

  cvCmpS(phi, 0, bw, CV_CMP_GT);
  icvInitializeLevelSet(bw,phi);

  // recognition with fourier descriptor
  if (initialized())
  {
    icvWarpInvert(warp_p, invwarp_p3x3);
    icvWarp(bw, bw_full, invwarp_p3x3);
    {
      CvMat * result = cvCreateMat(1,2,CV_32F);
      if (fdesc.predict(bw, result)){
        // cvPrintf(stderr, "%f,", result);
        m_status=(result->data.fl[0]>0.5);
      }
      cvReleaseMat(&result);
    }
  }

// cvShowImageEx("Test", imgYpatch, CV_CM_GRAY); CV_WAIT();
// cvShowImageEx("Test", Heaviside, CV_CM_GRAY); CV_WAIT();
// cvShowImageEx("Test", prterm1, CV_CM_GRAY); CV_WAIT();
  
  // learn new fg model and bg model
  // according to equation (8) in ECCV08 paper
  // if (m_t.m_framecounter>150)//(!has_fg)
  // if (1)
  if (initialized()) //  && iter2<=2
  {
    // const float alphaf = 0.0005f;
    // const float alphab = 0.0025f;
    // const float alphaf = 0.02f;
    // const float alphab = 0.025f;
    const float alphaf = 0.001f;
    const float alphab = 0.002f;

    int ystep=reffhist->dim[0].step/sizeof(float);
    int ustep=reffhist->dim[1].step/sizeof(float);
    int vstep=reffhist->dim[2].step/sizeof(float);
    assert(CV_MAT_TYPE(imgYpatch->type)==CV_8U);
    uchar * yptr=imgYpatch->data.ptr;
    uchar * uptr=imgUpatch->data.ptr;
    uchar * vptr=imgVpatch->data.ptr;
    float * phiptr = phi->data.fl;
    float * currptr=currhist->data.fl;
    float * reffptr=reffhist->data.fl;
    float * refbptr=refbhist->data.fl;
    int j,k,step=imgYpatch->step/sizeof(uchar);
    assert(step==nc);
    assert(step==bw->step/sizeof(uchar));
    assert(step==phi->step/sizeof(float));
    int ypos,upos,vpos,yupos;
    double currval,bthres=nc*0.1;
    assert((ystep==(1<<10))&&(ustep==(1<<5)));

    for (j=0;j<nr;j++)
    {
      for (k=0;k<nc;k++)
      {
        ypos=((yptr[k]>>3)<<10);
        upos=((uptr[k]>>3)<<5);
        vpos=vptr[k]>>3;
        yupos=ypos+upos;
        currval=(currptr+yupos)[vpos];
        if (phiptr[k]>(bthres))
        {
          (reffptr+yupos)[vpos]=
              (reffptr+yupos)[vpos]*(1.-alphaf)+currval*alphaf;
        }else if (phiptr[k]<(-bthres))
        {
          (refbptr+yupos)[vpos]=
              (refbptr+yupos)[vpos]*(1.-alphab)+currval*alphab;
        }
      }
      yptr+=step; uptr+=step; vptr+=step;
      phiptr+=step; 
    }
    cvScale(reffhist, reffhist, 1./cvSum(reffhist).val[0]);
    cvScale(refbhist, refbhist, 1./cvSum(refbhist).val[0]);
  }

  // cvShowImageEx("Pb",Pb,CV_CM_GRAY);
  
  // leave ONLY boundary of the mask
  // for (iter=0;iter<2;iter++)
  if (0)
  {
    uchar * ptr=bw->data.ptr;
    int j,k,step=bw->step/sizeof(uchar);
    for (j=0;j<nr-1;j++,ptr+=step)
      for (k=0;k<nc-1;k++)
      {
        ptr[k]=((ptr[k+1]!=ptr[k]) || ((ptr+step)[k]!=ptr[k])) ? 255:0;
      }    
  }

// #if 0
//   int nphis_totrain = 850;
//   static int nphis = 0;
//   static float * phimatnd =
//       (float*)malloc(nr*nc*sizeof(float)*nphis_totrain);
//   if ((nphis==nphis_totrain) && (phimatnd!=0))
//   {
//     // write to file
//     FILE * fp = fopen("../data/phimatnd.txt", "w");
//     fprintf(fp, "%d %d %d\n", nphis_totrain,nr,nc);
//     fwrite(phimatnd,sizeof(float),nr*nc*nphis_totrain,fp);
//     fclose(fp);
//     free(phimatnd);phimatnd=0;
//     CV_WAIT();
//   }else if (nphis<nphis_totrain){
//     // append new phi
//     memcpy(phimatnd+(nr*nc*nphis),phi->data.fl,sizeof(float)*nr*nc);
//     nphis++;
//   }
// #endif
  
  return 1;
}

//-------------------------------------------------------
// REGISTRATION
//-------------------------------------------------------
int CvPWPTracker::registration()
{
  // cvPrintf(stderr, "%f,", warp_p); CV_WAIT();
  int nr=phi->rows, nc=phi->cols;
  const int border = 5*m_iLinMultiplier;
  int iter;

  const int maxiter = 40;
  double rms_error=0;

  CvMat * B2 = cvCreateMat(nr, nc, CV_32F);
  CvMat * B1 = cvCreateMat(nr, nc, CV_32F);

  CvMat * Hv = Heaviside;

  // full size matrices
  if (!bw_full)
  { bw_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_8U); }
  if (!Pf_full)
  { Pf_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_32F); }
  if (!Pb_full)
  { Pb_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_32F); }
  if (!phi_full)
  { phi_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_32F); }

  // ---------------------------------
  // compute obj/bg ratio
  {
    cvCalcHeaviside(phi, Hv, 3.);
    cvCalcDirac(phi, dirac,  3.);
  }

  double etaf = cvSum(Heaviside).val[0];
  double etab = nc*nr-etaf;
  
  // initialize variable matrices with template size
  init_a(nr, nc); 

  //---------------------------------
  // step 3: evaluate the gradient of the template
  //---------------------------------
  // icvWarp(phi_full, phi_small, warp_p);
  // cvShowImageEx("Test", phi_small, CV_CM_GRAY); CV_WAIT();
  cvSobel(phi, img_dx, 1, 0, 1); cvScale(img_dx, img_dx, 0.5);
  cvSobel(phi, img_dy, 0, 1, 1); cvScale(img_dy, img_dy, 0.5);

  //---------------------------------
  // step 4: evaluate the Jacobian dW_dp
  //---------------------------------
  icvCalcStDescImages(dW_dp, img_dx, img_dy, VI_dW_dp);
  // cvShowImageEx("Test", VI_dW_dp, CV_CM_GRAY); CV_WAIT();

  // limit within a narrow band
  {
    assert((nc*N_p)==VI_dW_dp->cols);
    float * diracptr = dirac->data.fl;
    float * sdptr = VI_dW_dp->data.fl;
    int sdstep = VI_dW_dp->step/sizeof(float);
    int step = dirac->step/sizeof(float);
    int i,j,k;
    for (i=0; i<N_p; i++)
    {
      diracptr=dirac->data.fl;
      sdptr = VI_dW_dp->data.fl+nc*i;
      for (j=0; j<nr; j++){
// #if CV_PWP_USE_OPENMP
// #pragma omp parallel for shared(sdptr,diracptr) private(k)
// #endif
      for (k=0; k<nc; k++){
// #if CV_PWP_USE_NARROWBAND
//         if (diracptr[k]==0.0)
//         {sdptr[k]=0;}
//         else
// #endif // CV_PWP_USE_NARROWBAND
        {
          sdptr[k] = diracptr[k]*sdptr[k];
        }
      }
      sdptr+=sdstep; diracptr+=step;
      }
    }
  }
  // cvShowImageEx("Test", VI_dW_dp, CV_CM_GRAY); CV_WAIT();

  cvSet(Pf_full,cvScalar(-1));
  cvSet(Pb_full,cvScalar(-1));
  // single one-shoot pose estimation
  for (iter=0;iter<maxiter;iter++)
  {
    // step 2: compute the error image I(W(x;p))-T(x)
    rms_error=0;

    // -------------------------------------------------------
    // pixel-wise posteror - tracking section
    // -------------------------------------------------------
    {
      // cvShowImageEx("Test", Pf, CV_CM_GRAY); CV_WAIT();
      // cvShowImageEx("Test", Pb, CV_CM_GRAY); CV_WAIT();
      assert(CV_MAT_TYPE(B2->type)==CV_32F);
      assert(CV_MAT_TYPE(B1->type)==CV_32F);
      assert(CV_MAT_TYPE(VI_dW_dp->type)==CV_32F);
      assert(CV_MAT_TYPE(Hessian->type)==CV_64F);

      // icvWarp(Pf_full, Pf, warp_p);
      // icvWarp(Pb_full, Pb, warp_p);
      // icvWarpInvert(warp_p, invwarp_p3x3);
      // cvShowImageEx("Test", imgY, CV_CM_GRAY); CV_WAIT();
      icvCalcWarpPixelWisePosterior(
          imgY, imgU, imgV, reffhist, refbhist, etaf, etab,
          Pf_full, Pb_full, Pf, Pb, warp_p);
          // Pf_full, Pb_full, Pf, Pb, invwarp_p3x3);
      // cvShowImageEx("Test", Pf, CV_CM_GRAY); CV_WAIT();
      
      // compute matrix B2 & B1
      {
        int i,j;
        float * pfptr = Pf->data.fl;
        float * pbptr = Pb->data.fl;
        float * hvptr = Hv->data.fl;
        float * b2ptr = B2->data.fl;
        float * b1ptr = B1->data.fl;
        int step = nc;
        assert(step==Pf->step/sizeof(float));
        assert(step==Pb->step/sizeof(float));
        assert(step==Hv->step/sizeof(float));
        assert(step==B2->step/sizeof(float));
        assert(step==B1->step/sizeof(float));
        double pxval;
        for (i=0; i<nr; i++){
          for (j=0; j<nc; j++){
            pxval = hvptr[j]*pfptr[j]+(1.-hvptr[j])*pbptr[j];
            b1ptr[j]=
                (pfptr[j]-pbptr[j])/pxval;
            b2ptr[j]=b1ptr[j]*b1ptr[j];
            // b2ptr[j]=0.5*((pfptr[j]/(hvptr[j]*pxval))+
            //               (pbptr[j]/(pxval-pxval*hvptr[j])));
          }
          pfptr+=step; pbptr+=step; hvptr+=step;
          b2ptr+=step; b1ptr+=step;
        }
      }
// cvShowImageEx("Test", B2, CV_CM_GRAY); CV_WAIT();
// cvShowImageEx("Test", B1, CV_CM_GRAY); CV_WAIT();

      {
        cvZero(Hessian);
        int i,j,m,n;
        double * hptr = Hessian->data.db;
        float * j1ptr, * j2ptr, * b2ptr;
        int hstep = Hessian->step/sizeof(double);
        int jstep = VI_dW_dp->step/sizeof(float);
        int bstep = B2->step/sizeof(float);
        assert(CV_MAT_TYPE(B2->type)==CV_32F);
        assert( Hessian->cols==N_p );
        assert( hstep==N_p );
        for (i=0;i<N_p;i++)
        {
          for (j=0;j<N_p;j++)
          {
            j1ptr = VI_dW_dp->data.fl+nc*i;
            j2ptr = VI_dW_dp->data.fl+nc*j;
            b2ptr = B2->data.fl;
            for (m=0;m<nr;m++){
              for (n=0;n<nc;n++){
                hptr[j] += j1ptr[n]*j2ptr[n]*b2ptr[n];
              }
              j1ptr+=jstep;j2ptr+=jstep; b2ptr+=bstep;
            }
          }
          hptr+=hstep;
        }
      }
      // cvShowImageEx("Test", Hessian, CV_CM_GRAY); CV_WAIT();

      if (cvInvert(Hessian, H_inv, CV_LU)==0.0) {
        // lost
        m_initialized=0;m_status=-1;
        //CV_WAIT();//assert(false);
        cvInvert(Hessian, H_inv, CV_SVD);
      }
      // cvShowImageEx("Test", H_inv, CV_CM_GRAY); CV_WAIT();

      // step 7: update steepest descent images
      icvUpdateStDescImages(VI_dW_dp, B1, sd_delta_p);
      // cvPrintf(stderr, "%f,", sd_delta_p);

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
      if ( sqrt(dpsum) < 0.2 ) { break; }
      //if ( sqrt(dpsum) < 0.08 ) { break; }
    }

    // step 9: update the warp
    if (1)
    {
      icvWarpTranspose(delta_p, delta_M);

      // invert compositional warp
      if (cvInvert(delta_M, delta_M_inv, CV_LU)==0.0) {
        // lost
        m_initialized=0;m_status=-1;
        //CV_WAIT();//assert(false);
        cvInvert(delta_M, delta_M_inv, CV_SVD);
      }

      // current warp
      icvWarpReshape(warp_p, warp_M);

      // compose
      cvMatMul(warp_M, delta_M_inv, comp_M);
      icvWarpCompose(comp_M, warp_p);
    }

  } // end of iteration
  // fprintf(stderr, "registration[%d]\n", iter);
  // fprintf(stderr, "delta_p: %f\n",
  //         sqrt(pow(delta_p->data.db[0],2)+
  //              pow(delta_p->data.db[1],2)+
  //              pow(delta_p->data.db[2],2)+
  //              pow(delta_p->data.db[3],2)));
  // cvPrintf(stderr, "%f, ", delta_p);

  // ---------------------------------
  // drift correction
  {
    icvWarpInvert(warp_p, invwarp_p3x3);
    icvWarp(phi, phi_full, invwarp_p3x3);
    cvCmpS(phi_full, 1, bw_full, CV_CMP_GT);
  }
  icvWarp(bw_full, bw, warp_p);
  icvDriftCorrection(warp_p, bw, nr,nc,border,m_iLinMultiplier);
  // icvInitializeLevelSet(bw,phi);

  // ---------------------------------
  // convert warp_p to box representation
  if (1)
  {
    CvPoint2D32f pts[4]={0,};
    double angle = icvWarpToPoints(warp_p, pts, nr, nc);

    m_outerbox.angle = int(angle+720)%360;
    m_outerbox.center.x = (pts[0].x+pts[2].x)*0.5;
    m_outerbox.center.y = (pts[0].y+pts[2].y)*0.5;
    m_outerbox.size.width  =
        sqrt(pow(pts[1].x-pts[0].x,2)+pow(pts[1].y-pts[0].y,2));
    m_outerbox.size.height =
        sqrt(pow(pts[3].x-pts[0].x,2)+pow(pts[3].y-pts[0].y,2));
    // validation check
    if ( (m_outerbox.center.x<0||m_outerbox.center.x>m_imsize.width) ||
         (m_outerbox.center.y<0||m_outerbox.center.y>m_imsize.height) ||
         // (m_outerbox.size.width <9 ||m_outerbox.size.width >120) ||
         ( (m_outerbox.size.height<6*m_iLinMultiplier) ||
		   (m_outerbox.size.height>60*m_iLinMultiplier) ) )
    {
      m_initialized=0; m_status=-1;
    }
    m_innerbox = m_outerbox;
    m_innerbox.size.width  -= border*2.;
    m_innerbox.size.height -= border*2.;
  }

  if (B2) { cvReleaseMat(&B2); }
  if (B1) { cvReleaseMat(&B1); }

  return 1;
}

void icvDriftCorrection3(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier);
void icvDriftCorrection4(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier);
void icvDriftCorrection6(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier);

void icvDriftCorrection(CvMat * warp_p, CvMat * bw,
                        int nr, int nc, float border,
                        float multiplier)
{
  typedef void (*CvDriftCorrFuncType)(CvMat *, CvMat *,
                                      int,int,float,float);
  static CvDriftCorrFuncType driftcorrfuncarr[7] = {
    0,0,0,icvDriftCorrection3,icvDriftCorrection4,0,icvDriftCorrection6
  };
  assert(warp_p->cols==1);
  driftcorrfuncarr[warp_p->rows](warp_p,bw,nr,nc,border,multiplier);
}

void icvDriftCorrection3(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier)
{
  assert(warp_p->rows==3);
  int Bt=nr,Bb=-1,Bl=nc,Br=-1;
  {
    int i,j;
    uchar * bwptr=bw->data.ptr;
    int step=bw->step/sizeof(uchar);
    for (i=0;i<nr;i++){
      for (j=0;j<nc;j++){
        if (bwptr[j]){
          Bt=MIN(i,Bt); Bb=MAX(i,Bb);
          Bl=MIN(j,Bl); Br=MAX(j,Br);
        }
      }
      bwptr+=step;
    }
  }
  Bb=nr-Bb; Br=nc-Br;

  // fprintf(stderr, "border: %d,%d,%d,%d\n", Bt,Bb,Bl,Br);
  {
#if 1
    const float Lt = 0.4f*float(multiplier);
    const float Ls = 0.005f;
    float * pptr = warp_p->data.fl;
    // float scale = sqrt(pptr[0]*pptr[0])-1.;
    float scale = pptr[0]-1.;
    // pptr[0] = MAX(0.6,MIN(1.66,pptr[0]/(1.+MAX(-Ls,MIN(scale,Ls)))));
    pptr[0] = MAX(0.8,MIN(1.25,pptr[0]/(1.+MAX(-Ls,MIN(scale,Ls)))));
#else
    const float Lt = 0.4f*float(multiplier);
    const float Ls = 0.002f;
    float * pptr = warp_p->data.fl;
    float minB = MIN(MIN(MIN(Bt,Bl),Bb),Br);
    float scale = (border-minB)*0.005;
    pptr[0] = MAX(0.6,MIN(1.67,pptr[0]*(1.+MAX(-Ls,MIN(scale,Ls)))));
    // pptr[0] = pptr[0]*(1.+MAX(-Ls,MIN(scale,Ls)));
#endif
    pptr[1]+=MAX(-Lt,MIN(Bl-Br,Lt));
    pptr[2]+=MAX(-Lt,MIN(Bt-Bb,Lt));
  }
}

void icvDriftCorrection4(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier)
{
  assert(warp_p->rows==4);
  int Bt=nr,Bb=-1,Bl=nc,Br=-1;
  {
    int i,j;
    uchar * bwptr=bw->data.ptr;
    int step=bw->step/sizeof(uchar);
    for (i=0;i<nr;i++){
      for (j=0;j<nc;j++){
        if (bwptr[j]){
          Bt=MIN(i,Bt); Bb=MAX(i,Bb);
          Bl=MIN(j,Bl); Br=MAX(j,Br);
        }
      }
      bwptr+=step;
    }
  }
  Bb=nr-Bb; Br=nc-Br;

  // fprintf(stderr, "border: %d,%d,%d,%d\n", Bt,Bb,Bl,Br);
  {
    // const float Lt = 0.4*float(m_iLinMultiplier);
    const float Lt = 0.6*float(multiplier);
    // const float Ls = 0.1f;
    const float Ls = 0.005f;
    float * pptr = warp_p->data.fl;
    float minB = MIN(MIN(MIN(Bt,Bl),Bb),Br);
	float scaled = sqrt(pptr[0]*pptr[0]+pptr[1]*pptr[1]);
    float scale = (border-minB)*0.005;
#if 1
    pptr[0] = MAX(0.6*pptr[0]/scaled,       // lower bound
              MIN(1.67*pptr[0]/scaled,      // upper bound
                  pptr[0]*(1.+MAX(-Ls,MIN(scale,Ls)))));
    pptr[1] = MAX(0.6*pptr[1]/scaled,
              MIN(1.67*pptr[1]/scaled,
                  pptr[1]*(1.+MAX(-Ls,MIN(scale,Ls)))));
#else
    pptr[0] = pptr[0]*(1.+MAX(-Ls,MIN(scale,Ls)));
    pptr[1] = pptr[1]*(1.+MAX(-Ls,MIN(scale,Ls)));
#endif
    pptr[2] += MAX(-Lt,MIN(Bl-Br,Lt));
    pptr[3] += MAX(-Lt,MIN(Bt-Bb,Lt));
  }
}

void icvDriftCorrection6(CvMat * warp_p, CvMat * bw,
                         int nr, int nc, float border,
                         float multiplier)
{
  assert(warp_p->rows==6);
  int Bt=nr,Bb=-1,Bl=nc,Br=-1;
  {
    int i,j;
    uchar * bwptr=bw->data.ptr;
    int step=bw->step/sizeof(uchar);
    for (i=0;i<nr;i++){
      for (j=0;j<nc;j++){
        if (bwptr[j]){
          Bt=MIN(i,Bt); Bb=MAX(i,Bb);
          Bl=MIN(j,Bl); Br=MAX(j,Br);
        }
      }
      bwptr+=step;
    }
  }
  Bb=nr-Bb; Br=nc-Br;

  // fprintf(stderr, "border: %d,%d,%d,%d\n", Bt,Bb,Bl,Br);
  {
    const float Lt = 0.4f*float(multiplier);
    const float Ls = 0.005f;
    float * pptr = warp_p->data.fl;
    float scale = sqrt(pptr[0]*pptr[0]+pptr[1]*pptr[1])-1.;
    pptr[0]/=1.+MAX(-Ls,MIN(scale,Ls));
    pptr[1]/=1.+MAX(-Ls,MIN(scale,Ls));
    scale = sqrt(pptr[2]*pptr[2]+pptr[3]*pptr[3])-1.;
    pptr[2]/=1.+MAX(-Ls,MIN(scale,Ls));
    pptr[3]/=1.+MAX(-Ls,MIN(scale,Ls));
    pptr[4]+=MAX(-Lt,MIN(Bl-Br,Lt));
    pptr[5]+=MAX(-Lt,MIN(Bt-Bb,Lt));
  }
}

void icvFillRegion(CvMat * mask)
{
  // cvDilateEx(mask, 3);
  // cvErodeEx(mask, 4); // removing noises (experimental)

  // -------------------------------------------------------
  // reverse the mask to a new image
  int nr=mask->rows,nc=mask->cols;
  CvMat * revmask = cvCreateMat(nr, nc, CV_8U);
  int i,j;
  assert(CV_MAT_TYPE(mask->type)==CV_8U);
  uchar * mptr = mask->data.ptr;
  uchar * rptr = revmask->data.ptr;
  const int step = mask->step/sizeof(uchar);
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      rptr[j] = (mptr[j])?0:255;
    }
    mptr+=step;rptr+=step;
  }
  // -------------------------------------------------------
  // remove the mask 
  int iqsz = nr*nc;
  int * iq = (int*)malloc(sizeof(int)*iqsz);
  iq[0]=0;iq[1]=(nr-1)*nc+nc-1;iq[2]=(nr/2)*nc;iq[3]=(nr-1)*nc+nc/2;
  int iqf=0, iqb=4;
  int offset[4]={-nc,1,nc,-1};
  rptr = revmask->data.ptr;
  int ctr,nbr,ypos,xpos;
  // cvShowImage("Test", mask); CV_WAIT();
  // cvShowImage("Test", revmask); CV_WAIT();
  while ( iqf!=iqb && iqb  < iqsz) {
    ctr=iq[iqf++];
    // fprintf(stderr, "(%d %d): ", ctr/nc, ctr%nc);
    for (i=0;i<4;i++)
    {
      if ( (((ctr%nc)==0)&&(i==3)) || (((ctr%nc)==(nc-1))&&(i==1)) ) {
        continue;
      }
      nbr=ctr+offset[i];
      ypos=nbr/nc;xpos=nbr%nc;
      if ( (ypos>=0)&&(ypos<nr)&&(xpos>=0)&&(xpos<nc) )
      {
        // fprintf(stderr, "(%d %d), ", ypos, xpos);
        if (rptr[nbr]) { rptr[nbr]=0; iq[iqb++]=nbr; }
      }
    } // fprintf(stderr, "\n");
    // if (iqf%50==0) { cvShowImage("Test", revmask); CV_WAIT(); }
  }
  free(iq);
  // cvShowImage("Test", mask); CV_WAIT();
  // -------------------------------------------------------
  // apply the new mask
  mptr = mask->data.ptr;
  rptr = revmask->data.ptr;
  for (i=0;i<nr;i++){
    for (j=0;j<nc;j++){
      if (rptr[j]) {mptr[j]=255;}
      else {
        mptr[j]=(mptr[j])?255:0;
      }
    }
    mptr+=step;rptr+=step;
  }
  // cvShowImage("Test", mask); CV_WAIT();
  cvReleaseMat(&revmask);

  // cvDilateEx(mask, 1); // removing noises (experimental)
  // cvErodeEx(mask, 1);
}

void icvCalcHistFromMaskedRegion(
    CvMat * imgYpatch, CvMat * imgUpatch, CvMat * imgVpatch,
    CvMatND * reffhist, CvMatND * refbhist, CvMat * submask)
{
    cvZero(reffhist); cvZero(refbhist);
    uchar * maskptr=submask->data.ptr;
    uchar * yptr=imgYpatch->data.ptr;
    uchar * uptr=imgUpatch->data.ptr;
    uchar * vptr=imgVpatch->data.ptr;
    int ystep=reffhist->dim[0].step/sizeof(float),
        ustep=reffhist->dim[1].step/sizeof(float),
        vstep=reffhist->dim[2].step/sizeof(float);
    float * reffptr=reffhist->data.fl;
    float * refbptr=refbhist->data.fl;
    int j,k,step=imgYpatch->step/sizeof(uchar);
    int nr=imgYpatch->rows,nc=imgYpatch->cols;
    for (j=0;j<nr;j++)
    {
      for (k=0;k<nc;k++)
      {
        if (maskptr[k])
        {
          (reffptr+ystep*(yptr[k]>>3)+ustep*(uptr[k]>>3))[vptr[k]>>3] += 1;
        }
        else
        {
          (refbptr+ystep*(yptr[k]>>3)+ustep*(uptr[k]>>3))[vptr[k]>>3] += 1;
        }
      }
      yptr+=step;uptr+=step;vptr+=step;maskptr+=step;
    }
    // normalize histogram 
    cvScale(reffhist, reffhist, 1./cvSum(reffhist).val[0]);
    cvScale(refbhist, refbhist, 1./cvSum(refbhist).val[0]);
}

// extern void printminmax(CvMat *);
void icvCalcWarpPixelWisePosterior(
    CvMat * imgY, CvMat * imgU, CvMat * imgV,
    CvMatND * reffhist, CvMatND * refbhist, double etaf, double etab,
    CvMat * Pf_full, CvMat * Pb_full, CvMat * Pf, CvMat * Pb,
    CvMat * warp_p)
{
  int nr=Pf->rows,nc=Pf->cols;
  int i,j;

  double eta=etaf+etab;
  uchar * yptr = imgY->data.ptr;
  uchar * uptr = imgU->data.ptr;
  uchar * vptr = imgV->data.ptr;
  float * reffptr = reffhist->data.fl; 
  float * refbptr = refbhist->data.fl;
  float * pfptr = Pf->data.fl;
  float * pbptr = Pb->data.fl;
  float * pf2ptr = Pf_full->data.fl;
  float * pb2ptr = Pb_full->data.fl;
  float * pptr = warp_p->data.fl;
  int istep=imgY->step/sizeof(uchar);
  int dstep=Pf->step/sizeof(float);
  int ystep=reffhist->dim[0].step/sizeof(float);
  int ustep=reffhist->dim[1].step/sizeof(float);
  int vstep=reffhist->dim[2].step/sizeof(float);
  // float cp0=pptr[0], sp1=pptr[1], sp3=pptr[3], cp4=pptr[4];
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  if ((warp_p->cols==1)&&(warp_p->rows==4)){
    cp0=pptr[0]; sp1=-pptr[1]; sp3=pptr[1]; cp4=pptr[0];
  }else if ((warp_p->cols==1)&&(warp_p->rows==3)){
    cp0=pptr[0]; sp1=0; sp3=0; cp4=pptr[0];
  }else{
    fprintf(stderr, "WARNING: unknown warp parameter setting !\n");
  }
  int ww=imgY->width-1, hh=imgY->height-1;
  int ypos, upos, vpos, yupos;
  CvPoint pos; double reffval,refbval,mfval,mbval,pfval,pbval,tmp;

  assert(warp_p->cols==1);
  int txid=warp_p->rows-2,tyid=warp_p->rows-1;
  if (warp_p->rows==3) {sp1=sp3=0;}

  assert(istep==imgU->step/sizeof(uchar));
  assert(istep==imgV->step/sizeof(uchar));
  assert(istep==Pf_full->step/sizeof(float));
  assert(istep==Pb_full->step/sizeof(float));
  assert(dstep==Pb->step/sizeof(float));

  for (i=0;i<nr;i++)
  {
    for (j=0;j<nc;j++)
    {
      // pos.x = cvRound(j*cp0+i*sp1+pptr[2]); // nearest-neighbor method
      // pos.y = cvRound(j*sp3+i*cp4+pptr[5]);
      pos.x = cvRound(j*cp0+i*sp1+pptr[txid]); // nearest-neighbor method
      pos.y = cvRound(j*sp3+i*cp4+pptr[tyid]);

      if ( (pos.x>ww) || (pos.x<0) || (pos.y>hh) || (pos.y<0) )
      {
        pfptr[j] = 1e-5;
        pbptr[j] = 0.99;//1./etab; //wptr[x]=1e-5;
      }else{
        pfval = (pf2ptr+istep*pos.y)[pos.x];
        pbval = (pb2ptr+istep*pos.y)[pos.x];
        if ((pfval<0)||(pbval<0))
        {
          ypos = int((yptr+istep*pos.y)[pos.x])>>3;
          upos = int((uptr+istep*pos.y)[pos.x])>>3;
          vpos = int((vptr+istep*pos.y)[pos.x])>>3;
          yupos = ystep*ypos+ustep*upos;
          reffval=(reffptr+yupos)[vpos];
          refbval=(refbptr+yupos)[vpos];
          // compute {Mf, Mb}
          tmp = 1./(reffval+refbval+1e-5);
          mfval=MAX(1e-5,reffval*tmp);
          mbval=MAX(1e-5,refbval*tmp);
          // compute {Pf, Pb}
          tmp=1./(etaf*mfval+etab*mbval);
          pfptr[j]=MAX(1E-5,mfval*tmp);
          pbptr[j]=MAX(1E-5,mbval*tmp);
          (pf2ptr+istep*pos.y)[pos.x] = pfptr[j];
          (pb2ptr+istep*pos.y)[pos.x] = pbptr[j];
        }else{
          pfptr[j] = pfval;
          pbptr[j] = pbval;
        }
      }
    }
    pfptr+=dstep;pbptr+=dstep;
  }

  // printminmax(Pf);printminmax(Pb);
  // CV_SHOW(Pf);CV_SHOW(Pb);CV_SHOW(Pf_full);CV_SHOW(Pb_full);
  // CV_SHOW(Pb);
}

//////////////////////////////////////////////////////////////////////////

#if !defined(ANDROID) && defined(WITH_TZG)

//#include "LevelsetTracking.h"
////#include "ShowTestImage.h"
//
#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#ifndef PI
#define PI 3.1415926
#endif
//
//CLevelsetTracking myLevelsetTk;

int CvPWPTracker::myInitialize(CvRect roi)
{
	m_initialized=0;
	copy_to_local();

	int imgWidth = m_t.m_currColorImage->width;
	int imgHeight = m_t.m_currColorImage->height;

	static const float bbb=6.*m_iLinMultiplier;
	m_outerroi = roi;
	m_outerroi.x = max(0, roi.x-bbb);
	m_outerroi.y = max(0, roi.y-bbb);
	m_outerroi.width  = min(imgWidth - 1, roi.width +bbb*2);
	m_outerroi.height = min(imgHeight - 1, roi.height+bbb*2);
	if ( (m_outerroi.x<0) || (m_outerroi.y<0) ||
		((m_outerroi.width+m_outerroi.x)>m_imsize.width) ||
		((m_outerroi.height+m_outerroi.y)>m_imsize.height) )
	{
		fprintf(stderr, "WARNING: fail to initialize - on the boundary\n");
		return 0;
	} // fail to initialize - on the boundary
	//if(m_outerroi.width > imgWidth/3 || m_outerroi.height > imgHeight/3)
	//{
	//	return 0;
	//}

	m_outerbox = cvBox2DFromRect(m_outerroi);

	//{
	//	CvPoint2D32f pts[4];
	//	cvBoxPoints32f(m_outerbox, pts);
	//	icvWarpInit(warp_p,pts[0].x,pts[0].y);
	//}

	CvBox2D box = m_outerbox;

	//assert(phi==NULL);
	// ---------------------------------
	// calculate histogram of current frame
	{
		CvMat imgY_stub,imgU_stub,imgV_stub;
		CvMat submask_stub,*submask;
		// box.size.width =MIN(24*m_iLinMultiplier,box.size.width);
		// box.size.height=MIN(30*m_iLinMultiplier,box.size.height);
		box.size.width =MIN(30*m_iLinMultiplier, box.size.width);
		box.size.height=MIN(36*m_iLinMultiplier, box.size.height);
		CvRect roiex = m_outerroi;
		//assert( phi==NULL && bw==NULL);
		phi = cvCreateMat(roiex.height, roiex.width, CV_32F);
		bw  = cvCreateMat(roiex.height, roiex.width, CV_8U);
		if (!imgYpatch)
		{ imgYpatch = cvCloneMat(cvGetSubRect(imgY, &imgY_stub, roiex)); }
		if (!imgUpatch)
		{ imgUpatch = cvCloneMat(cvGetSubRect(imgU, &imgU_stub, roiex)); }
		if (!imgVpatch) 
		{ imgVpatch = cvCloneMat(cvGetSubRect(imgV, &imgV_stub, roiex)); }
		submask   = cvCloneMat(cvGetSubRect(silhImage, &submask_stub, roiex));

		//cvShowImage("Test", submask); CV_WAIT();
		icvFillRegion(submask);
		// cvShowImage("Test", submask); CV_WAIT();
		// calculate histogram of current frame
		//icvCalcHistFromMaskedRegion(
		//	imgYpatch, imgUpatch, imgVpatch, reffhist, refbhist, submask);
		//// initialize phi !~
		//icvInitializeLevelSet(submask, phi);

		// ---------------------------------
		// additional segmentation steps
		// for accurate initial pixel-wise posterior
		//for (int i=0;i<5;i++)
		//{
		//	// initialize segmentation !~
		//	segmentation();
		//	cvCmpS(Heaviside,0.9,submask,CV_CMP_GT);
		//	// calculate histogram of current frame
		//	icvCalcHistFromMaskedRegion(
		//		imgYpatch, imgUpatch, imgVpatch, reffhist, refbhist, submask);
		//	// initialize phi !~
		//	icvInitializeLevelSet(submask, phi);
		//}
		//cvShowImage("Test", submask); CV_WAIT();

		//////////////////////////////////////////////////////////////////////////
		MobiRect roiRect;
		//roiRect.xMin = roi.x - 5;
		//roiRect.xMax = roi.x + roi.width + 10;
		//roiRect.yMin = roi.y;
		//roiRect.yMax = roi.y + roi.height;
		roiRect.xMin = roi.x;
		roiRect.xMax = roi.x + roi.width;
		roiRect.yMin = roi.y ;//- 10;
		roiRect.yMax = roi.y + roi.height;// + 20;
		IplImage* imgYUV = m_t.m_currColorImage;
		IplImage* imgMask = cvCreateImage(cvGetSize(imgY), 8, 1);
		cvZero(imgMask);
		CvMat submat;
		cvGetSubRect(imgMask, &submat, m_outerroi);
		cvCopy(submask, &submat);	

//cvShowTestImage(imgMask);
//cvWaitKey();
//
//		//refine mask??
//		CLsMaskProcess maskProc;
//		maskProc.CreateModel(imgYUV, imgMask, roiRect);
//		maskProc.RefineMask(imgYUV, imgMask, roiRect);
//
//cvShowTestImage(imgMask);
//cvWaitKey();

		myLevelsetTk.Init(imgYUV, imgMask, roiRect);

		//m_lkTrack.Init();

		//myLevelsetTk.RefineTemplate(imgYUV, imgMask);
//cvRectangle(imgMask, cvPoint(roiRect.xMin,roiRect.yMin), cvPoint(roiRect.xMax, roiRect.yMax), cvScalar(255,255,255));
//cvShowTestImage(imgMask);
//cvWaitKey();

		cvReleaseImage(&imgMask);
		//////////////////////////////////////////////////////////////////////////

		// release memory
		cvReleaseMat(&imgYpatch); imgYpatch=NULL;
		cvReleaseMat(&imgUpatch); imgUpatch=NULL;
		cvReleaseMat(&imgVpatch); imgVpatch=NULL;
		cvReleaseMat(&submask);
	}

	//IplImage* phi = myLevelsetTk.getPhi();
	//REALT* pPhi = (REALT*)phi->imageData;
	//int sumFg = 0;
	//for(int y=0; y<phi->height; ++y)
	//{
	//	for(int x=0; x<phi->width; ++x)
	//	{
	//		if(pPhi[x] > 0)
	//		{
	//			++sumFg;
	//		}		
	//	}
	//	pPhi += myLevelsetTk.m_step;
	//}
	//if( sumFg < 0.7*phi->height*phi->width)
	//{
	//	return 0;
	//}

	m_initialized=1;
	m_updateFrameNum = 0;

	return 1;
}

int CvPWPTracker::myRegistration()
{
#if 0
	//////////////////////////////////////////////////////////////////////////
	////fast movement
	//IplImage* imgMask = cvCreateImage(cvGetSize(imgY), 8, 1);
	//cvCopy(silhImage, imgMask);	
	//addSkinOffset(m_t.m_currColorImage, imgMask);
	//cvReleaseImage(&imgMask);
	addSkinOffset(0, 0);
	//////////////////////////////////////////////////////////////////////////
#endif
CV_TIMER_START();
	IplImage* imgYUV = m_t.m_currColorImage;
	myLevelsetTk.Registrate(imgYUV, 10);
CV_TIMER_SHOW();
	return 1;
}


int CvPWPTracker::mySegmentation()
{
CV_TIMER_START();
	IplImage* imgYUV = m_t.m_currColorImage;
	myLevelsetTk.Segmentation(imgYUV, 1);

#if 1
	//if(m_updateFrameNum < 100)
	{
		IplImage* imgMask = cvCreateImage(cvGetSize(imgY), 8, 1);
		cvCopy(silhImage, imgMask);

		//refine mask??		
		//CLsMaskProcess maskProc;
		//MobiRect roiRect;
		//myLevelsetTk.getMaxfgBoundingbox(imgMask, roiRect.xMin, roiRect.yMin, roiRect.xMax, roiRect.yMax);		
		//maskProc.CreateModel(imgYUV, imgMask, roiRect);
		//maskProc.RefineMask(imgYUV, imgMask, roiRect);


		myLevelsetTk.Update(imgYUV, imgMask);
		cvReleaseImage(&imgMask);
		
		//++m_updateFrameNum;
	}
#endif
CV_TIMER_SHOW();


	return 1;
}

int CvPWPTracker::myRecognition()
{
	//use my level set scale change
	//float* pW = myLevelsetTk.getW()->data.fl;

	//REALT scale = sqrt(pW[0]*pW[0] + pW[1]*pW[1]);

	//FILE* fp = fopen("d:/scale.txt", "a");
	//fprintf(fp, "%f\n", scale);
	//fclose(fp);

	//static deque<REALT> scaleRecord;

	//int numRec = (int)scaleRecord.size();
	//const int numMaxRecord = 15;

	////int 
	//static REALT mu = 0, sigma = 0;

	//if(numRec < numMaxRecord)
	//{
	//	//mu = (mu*numRec + scale)/(numRec+1);
	//	//sigma = sigma + (scale - mu)*(scale - mu);

	//	scaleRecord.push_back(scale);
	//}
	//else
	//{
	//	//mu = mu*numMaxRecord -  scaleRecord[0] + scale

	//	scaleRecord.pop_front();
	//	scaleRecord.push_back(scale);

	//	
	//}

	

#if 1
#if 1
	CLevelsetTracking& lsTk = myLevelsetTk;
	CvMat * phi = cvCreateMat(lsTk.m_modelHeight, lsTk.m_modelWidth, CV_32F);
	cvCopy(lsTk.getPhi(), phi);

	if (!prior.initialized())
	{
		prior.initialized();
	}

	CvMat * prterm0 = prior.shapeterm0(phi);
	CvMat * prterm1 = prior.shapeterm1(phi); 

	//fprintf(stderr, "status: %f\n", prior.proj->data.fl[0]);
	m_status =  prior.status();//prior.proj->data.fl[0]>(-50.f);//
	cvReleaseMat(&phi);	

#else
	CvMat * result = cvCreateMat(1,2,CV_32F);
	CvMat * bw = cvCreateMat(myLevelsetTk.m_modelHeight, myLevelsetTk.m_modelWidth, CV_8U );
	cvCmpS(myLevelsetTk.getPhi(), 0, bw, CV_CMP_GT);
	if (fdesc.predict(bw, result))
	{
		fprintf(stderr, "status: %f\n", result->data.fl[0]);

		m_status=!(result->data.fl[0]>0.5);		
	}
	int aaa=0;
	aaa;
	if (aaa){cvSaveImage("d:/bw.bmp", bw);}

	cvReleaseMat(&result);
	cvReleaseMat(&bw);
#endif
#endif

	return 0;
}

int CvPWPTracker::myDisplay(IplImage* imgTest)
{
#if 0
	CLevelsetTracking& lsTk = myLevelsetTk;
	if(lsTk.getPhi() == 0)
	{
		return 0;
	}

	CvMat * phi_full = 0;
	CvMat * phi = 0;
	if (!phi_full)
	{ phi_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_32F); }

	if (!phi)
	{ phi = cvCreateMat(lsTk.m_modelHeight, lsTk.m_modelWidth, CV_32F); }
	cvCopy(lsTk.getPhi(), phi);

	CvMat* invwarp_p3x3 = cvCreateMat(3,3, CV_32F);
	CvMat * bw_full = 0;
	if (!bw_full)
	{ bw_full = cvCreateMat(m_imsize.height, m_imsize.width, CV_8U); }

	//icvWarpInvert(lsTk.getW(), invwarp_p3x3);
	cvInvert(lsTk.getW(), invwarp_p3x3);
	icvWarp(phi, phi_full, invwarp_p3x3);
	cvCmpS(phi_full, 0.01, bw_full, CV_CMP_GT);

	unsigned char* pTest = (unsigned char*)imgTest->imageData;
	uchar * ptr = bw_full->data.ptr;
	int j,k,step= bw_full->step/sizeof(uchar);
	for(int y=0; y<m_imsize.height-1; ++y, ptr+=step, pTest += imgTest->widthStep)
	{
		for(int x=0; x<m_imsize.width-1; ++x)
		{
			if(ptr[x] != ptr[x+1] || ptr[x] != (ptr+step)[x])
			{
				cvLine(imgTest, cvPoint(x, y), cvPoint(x, y), CV_RGB(255,255,255));
			}
		}		
	}

	cvReleaseMat(&invwarp_p3x3);
	cvReleaseMat(&phi_full);
	cvReleaseMat(&phi);
	cvReleaseMat(&bw_full);

	REALT* pPhi = (REALT*)lsTk.getPhi()->imageData;
	REALT x1, y1, x2, y2;
	CvScalar color = CV_BLUE;
	int bold = 1;
	lsTk.WarpingPoint(lsTk.getW(), 0, 0, x1, y1);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, 0, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, lsTk.m_modelHeight, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0, lsTk.m_modelHeight, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0, 0, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);

	bold = 2;
	int sz = -lsTk.m_sideSize;
	lsTk.WarpingPoint(lsTk.getW(), 0-sz, 0-sz, x1, y1);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth+sz, 0-sz, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth+sz, lsTk.m_modelHeight+sz, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0-sz, lsTk.m_modelHeight+sz, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0-sz, 0-sz, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
#else
	//IplImage* imgYUV = m_t.m_currColorImage;
	CLevelsetTracking& lsTk = myLevelsetTk;
	//IplImage* imgTest = cvCloneImage(imgYUV);
	
	if(lsTk.getPhi() == 0)
	{
		return 0;
	}
	if(!m_initialized)
	{
		return 0;
	}

	REALT* pPhi = (REALT*)lsTk.getPhi()->imageData;
	REALT x1, y1, x2, y2;
	CvScalar color = CV_BLUE;
	int bold = 1;
	lsTk.WarpingPoint(lsTk.getW(), 0, 0, x1, y1);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, 0, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, lsTk.m_modelHeight, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0, lsTk.m_modelHeight, x2, y2);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	lsTk.WarpingPoint(lsTk.getW(), 0, 0, x1, y1);
	cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);

	//bold = 2;
	//int sz = -lsTk.m_sideSize;
	//lsTk.WarpingPoint(lsTk.getW(), 0-sz, 0-sz, x1, y1);
	//lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth+sz, 0-sz, x2, y2);
	//cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	//lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth+sz, lsTk.m_modelHeight+sz, x1, y1);
	//cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	//lsTk.WarpingPoint(lsTk.getW(), 0-sz, lsTk.m_modelHeight+sz, x2, y2);
	//cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);
	//lsTk.WarpingPoint(lsTk.getW(), 0-sz, 0-sz, x1, y1);
	//cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x2+0.5, y2+0.5), color, bold);

	color = CV_RED;
	for(int y=0; y<lsTk.m_modelHeight-1; ++y)
	{
		for(int x=0; x<lsTk.m_modelWidth-1; ++x)
		{
			if(pPhi[x]*pPhi[x+1] < 0 || (pPhi+lsTk.m_step)[x]*pPhi[x] < 0)
			{
				REALT x1, y1;
				lsTk.WarpingPoint(lsTk.getW(), (REALT)x, (REALT)y, x1, y1);
				cvLine(imgTest, cvPoint(x1+0.5, y1+0.5), cvPoint(x1+0.5, y1+0.5), color);
			}		
		}
		pPhi += lsTk.m_step;
	}
	//cvShowTestImage(imgTest);
	//cvWaitKey(1);
	//cvReleaseImage(&imgTest);
#endif

	return 1;
}

void CvPWPTracker::checkTrackFalse()
{

	CLevelsetTracking& lsTk = myLevelsetTk;

	if(lsTk.getW() == 0)
	{
		return;
	}

	int numOutlier = 0;

	REALT x1, y1, x2, y2;
	lsTk.WarpingPoint(lsTk.getW(), 0, 0, x1, y1);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, 0, x2, y2);
	if(x1 < 0 || y1 < 0 || x1 >= m_imsize.width || y1 >= m_imsize.height)
	{
		++numOutlier;
	}
	if(x2 < 0 || y2 < 0 || x2 >= m_imsize.width || y2 >= m_imsize.height)
	{
		++numOutlier;
	}
	double dx = x2 - x1;
	double dy = y2 - y1;
	double warpWidth = sqrt(dx*dx + dy*dy);
	lsTk.WarpingPoint(lsTk.getW(), lsTk.m_modelWidth, lsTk.m_modelHeight, x1, y1);
	if(x1 < 0 || y1 < 0 || x1 >= m_imsize.width || y1 >= m_imsize.height)
	{
		++numOutlier;
	}
	dx = x2 - x1;
	dy = y2 - y1;
	double warpHeight = sqrt(dx*dx + dy*dy);
	int wthre = max(m_t.m_currColorImage->width, m_t.m_currColorImage->height)>>1;
	int wthre2 = lsTk.m_modelWidth*lsTk.m_modelHeight*0.2;
	//int wthre3 = lsTk.m_modelWidth*lsTk.m_modelHeight*0.99;
	
	lsTk.WarpingPoint(lsTk.getW(), 0, lsTk.m_modelHeight, x1, y1);
	if(x1 < 0 || y1 < 0 || x1 >= m_imsize.width || y1 >= m_imsize.height)
	{
		++numOutlier;
	}

	if(warpWidth < 8 || warpHeight < 8
		||
		warpWidth > wthre
		|| warpHeight > wthre
		//|| lsTk.m_numImage < wthre2
		//|| lsTk.m_numImage > wthre3
		|| numOutlier > 2
		)
	{
		m_initialized = 0;
	}

}
//
//#include "bbox.h"
//void CvPWPTracker::addSkinOffset(IplImage* imgRGB, IplImage* imgSil)
//{
//#if 1
//	CLevelsetTracking& lsTk = myLevelsetTk;
//	REALT xHand, yHand, su;
//	lsTk.getTrackingCenter(xHand, yHand, su);
//
//	REALT suHand = max(5, min(20, su*0.3));
//
//	myRect BB1;//last frame object bounding box
//	BB1.left = max(0, xHand - suHand);
//	BB1.right = min(m_imsize.width - 1, xHand + suHand);
//	BB1.top = max(0, yHand - suHand);
//	BB1.bottom = min(m_imsize.height - 1, yHand + suHand);
//
//	//generate 10x10 grid of points within BB1 with margin 5 px
//	CvMat* xFI = 0;
//	bb_points(xFI, BB1, 10, 10, 5);
//
//	if(!xFI)
//	{
//		return;
//	}
//
//	IplImage* imgLast = m_t.m_prevImage;
//	IplImage* imgCurr = cvCreateImage(cvGetSize(imgLast), 8, 1);
//	cvCopy(m_t.m_currImage, imgCurr);
//
//	CvMat* xFJ = 0;
//	m_lkTrack.track(xFJ, imgLast, imgCurr, xFI, xFI);
//
//	cvReleaseImage(&imgCurr);
//
//	if(!xFJ)
//	{
//		cvReleaseMat(&xFI);
//		return;
//	}
//
//	//get median of Forward-Backward error
//	double medFB = getMedian2(xFJ, 2);
//	
//	//too unstable predictions
//	if(medFB > 10)
//	{	
//		cvReleaseMat(&xFI);
//		cvReleaseMat(&xFJ);	
//		return;
//	}
//
//	//get median for NCC
//	double medNCC = getMedian2(xFJ, 3);
//
//	//get indexes of reliable points
//	vector<int> idxF;
//	double *pData3 = xFJ->data.db + 2*xFJ->cols;
//	double *pData4 = xFJ->data.db + 3*xFJ->cols;
//	for(int i=0; i<xFJ->cols; ++i)
//	{
//		if(pData3[i] <= medFB && pData4[i] >= medNCC)
//		{
//			idxF.push_back(i);
//		}		
//	}
//
//	//estimate BB2 using the reliable points only
//	int nselect = (int)idxF.size();
//
//	if(nselect == 0)
//	{
//		cvReleaseMat(&xFI);
//		cvReleaseMat(&xFJ);		
//
//		return;
//	}
//
//	CvMat* xFI_sel = cvCreateMat(2, nselect, CV_64FC1);
//	CvMat* xFJ_sel = cvCreateMat(2, nselect, CV_64FC1);
//	double* pxFISel = xFI_sel->data.db;
//	double* pxFJSel = xFJ_sel->data.db;
//	for(int i=0; i<nselect; ++i)
//	{
//		int idx = idxF[i];
//
//		pxFISel[i] = xFI->data.db[idx];
//		(pxFISel+nselect)[i] = (xFI->data.db+xFI->cols)[idx];
//
//		pxFJSel[i] = xFJ->data.db[idx];
//		(pxFJSel+nselect)[i] = (xFJ->data.db+xFJ->cols)[idx];
//	}
//
//	myRect BB2    = bb_predict(BB1,xFI_sel,xFJ_sel);
//
//	REALT dx = (BB2.left + BB2.right - BB1.left - BB1.right)*0.5;
//	REALT dy = (BB2.top + BB2.bottom - BB1.top - BB1.bottom)*0.5;
//
//	lsTk.addOffset(dx, dy);
//
//	cvReleaseMat(&xFI);
//	cvReleaseMat(&xFJ);
//	cvReleaseMat(&xFI_sel);
//	cvReleaseMat(&xFJ_sel);
//	//cvReleaseMat(&patchJ);
//#else
//	IplImage* hsv = cvCreateImage(cvGetSize(imgRGB), 8, 3);
//IplImage* imgSkin = cvCreateImage(cvGetSize(imgRGB), 8, 1);
//cvZero(imgSkin);
//
//	cvCvtColor(imgRGB, hsv, CV_BGR2HSV);
//
//	CLevelsetTracking& lsTk = myLevelsetTk;
//	
//
//	REALT xHand, yHand, su;
//	lsTk.getTrackingCenter(xHand, yHand, su);//(imgSil, xMin, yMin, xMax, yMax);
//
//	int size = su*0.6;
//
//	int xMin = max(0, xHand - size);
//	int xMax = min(m_imsize.width-1, xHand + size);
//	int yMin = max(0, yHand - size);
//	int yMax = min(m_imsize.height - 1, yHand + size);
//
//
//	int m00 = 0;
//	int m01 = 0;
//	int m10 = 0;
//
//	unsigned char* pBG = (unsigned char*)imgSil->imageData + yMin*imgSil->widthStep;
//	unsigned char* pHsv = (unsigned char*)hsv->imageData + yMin*hsv->widthStep;
//unsigned char* pSkin = (unsigned char*)imgSkin->imageData + yMin*imgSkin->widthStep;
//
//	for(int y=yMin; y<yMax; ++y, 
//		pBG+=imgSil->widthStep, pHsv += hsv->widthStep)
//	{
//		for(int x = xMin; x<xMax; ++x)
//		{
//			if(pBG[x] > 0)
//			{
//				int x3 = x+x+x;
//				int h = pHsv[x3];
//				int s = pHsv[x3 + 1];
//				int v = pHsv[x3 + 2];
//				//inRange(hsv, Scalar(0, 58, 89), Scalar(25, 173, 229), bw);
//				if( h>0 && h<25 && s>58 && s<173 && v>89 && v<229)
//				{
//pSkin[x] = 1;
//
//					++m00;
//					m01 += y;
//					m10 += x;
//				}
//			}
//		}
//pSkin+= imgSkin->widthStep;
//	}
////cvNamedWindow("skin", 1);
////cvScale(imgSkin, imgSkin, 255);
////cvShowImage("skin", imgSkin);
////cvWaitKey(0);
//cvReleaseImage(&imgSkin);
//
//	cvReleaseImage(&hsv);
//	
//
//	if(m00 > 0)
//	{
//		float m00inv = 1.0f/m00;
//		float xc = m10*m00inv;
//		float yc = m01*m00inv;
//
//		float dx = xc - xHand;
//		float dy = yc - yHand;
//
//		lsTk.addOffset(dx, dy);
//	}
//#endif
//}

#endif // ANDROID
