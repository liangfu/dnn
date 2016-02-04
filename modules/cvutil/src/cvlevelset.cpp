/**
 * @file   cvlevelset.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Dec 11 14:05:28 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvlevelset.h"

void icvInitializeLevelSet(CvMat * mask, CvMat * phi)
{
  assert(mask->cols==phi->cols);
  assert(mask->rows==phi->rows);
#if 0
  // initialize phi with smoothed mask
  cvSet(phi, cvScalar(phi->rows));
  cvSet(phi, cvScalar(-phi->rows), mask);
  {
    int ksize = mask->cols/4*2+1;
    cvSmooth(phi, phi, CV_GAUSSIAN, ksize, ksize);
  }
#else
  // initialize phi with distance transform
  int nr=mask->rows,nc=mask->cols;
  assert(CV_MAT_TYPE(phi->type)==CV_32F);
  CvMat * distmap = cvCreateMat(nr, nc, CV_32F);
  CvMat * tmp = cvCreateMat(nr, nc, CV_8U);
  cvCmpS(mask,0,tmp,CV_CMP_EQ);
  cvDistTransform(tmp,phi);
  cvDistTransform(mask,distmap);
  cvSub(distmap,phi,phi);
  cvReleaseMat(&tmp);
  cvReleaseMat(&distmap);
#endif
}

//--------------------------------
// fast sweep non-zero region within given ROI and the seed, output
// collected points into queue sequence.
// Example:
//   CvMemStorage * storage = cvCreateMemStorage();
//   // .. generate contour here ..
//   CvSeq * seq = cvCreateSeq(
//     CV_32SC2,            // sequence of integer elements
// 	   sizeof(CvSeq),       // header size - no extra fields
// 	   sizeof(CvPoint),     // element size
// 	   storage /* the container storage */ );
//   CvPoint seed = cvPoint(12,12);
//   cvSweep(binary_image, cvRect(10,10,100,100), seed, seq);
//   ...
// 
// imgSrc : source image
// roi : region of interest
// pt : seed point
// qSeq : queue, as list of sequence points
// return value: area of the propagated region
void cvSweep(CvArr * src, CvRect roi, CvPoint pt, 
             CvSeq * qSeq ) // queue
{
  IplImage header; CvMat regionhdr;
  IplImage * imgSrc = cvGetImage(src, &header);

  static const int nDx[4] = {0,1,0,-1};
  static const int nDy[4] = {-1,0,1,0};
  int m_imgSize = imgSrc->width*imgSrc->height;
  int xMin, xMax, yMin, yMax;
  xMin = roi.x; xMax = roi.x+roi.width;
  yMin = roi.y; yMax = roi.y+roi.height;
  int nCurrX=xMin, nCurrY=yMin;
  int idxCurr;
  int nStart=0, nEnd=1; // position in Queue
  cvClearSeq(qSeq);
  {
    CvPoint pt2=cvPoint(pt.x, pt.y);
    cvSeqPush(qSeq, &pt2);
  }

  // position of the 4 connected point
  int xx = 0;
  int yy = 0;
  int k = 0;

  int offset[4] = {
    imgSrc->widthStep*nDy[0] + nDx[0], // U
    imgSrc->widthStep*nDy[1] + nDx[1], // R
    imgSrc->widthStep*nDy[2] + nDx[2], // D
    imgSrc->widthStep*nDy[3] + nDx[3]  // L
  };

  uchar * pDataSrc = (unsigned char*)imgSrc->imageData;
  while (nStart<nEnd && nEnd<m_imgSize)
  {
    // position of the current seed
    nCurrX = CV_GET_SEQ_ELEM(CvPoint, qSeq, nStart)->x;
	nCurrY = CV_GET_SEQ_ELEM(CvPoint, qSeq, nStart)->y;
	idxCurr = -1; 

    // Search the 4 connected point of current seed
    for (k=0; k<4; k++) 
    {    
      xx = nCurrX + nDx[k];
      yy = nCurrY + nDy[k];
      if (xx<xMin || xx>xMax || yy<yMin || yy>yMax)
      {
        continue; // not in range of ROI
      }
      if (idxCurr<0)
      {
        idxCurr = imgSrc->widthStep * nCurrY + nCurrX;
      }

      //if ( pDataSrc[ idxCurr + offset[k] ] == 1 ) 
	  if ( pDataSrc[ idxCurr + offset[k] ] != 0 ) 
      {
        // pixel in (xx,yy) to stack
        CvPoint pt=cvPoint(xx, yy);
	    cvSeqPush(qSeq, &pt);
        pDataSrc[ idxCurr + offset[k] ] = 0;

        ++nEnd;  // Stack end point move forward
        if (nEnd>=m_imgSize)
        {
          break;
        }
      }   
    }
    ++nStart;
  }
}

void cvSeededBinaryDilate(CvArr * _img, CvPoint seed, const int dist)
{
  IplImage imghdr;
  IplImage * imgSrc = cvGetImage(_img,&imghdr); 

  // set all non-zero pixels to ONE for as binary mask
  cvSet(imgSrc, cvScalar(1), imgSrc); 

  static const int nDx[4] = {0,1,0,-1};
  static const int nDy[4] = {-1,0,1,0};
  int m_imgSize = imgSrc->width*imgSrc->height;
  int xMin, xMax, yMin, yMax;
  xMin = 0; xMax = imgSrc->width;
  yMin = 0; yMax = imgSrc->height;
  int nCurrX=xMin, nCurrY=yMin;
  int idxCurr;
  int nStart=0, nEnd=1; // position in Queue
  static CvPoint pts[19200];pts[0]=seed;

  // position of the 4 connected point
  int xx = 0;
  int yy = 0;
  int k = 0;

  int offset[4] = {
    imgSrc->widthStep*nDy[0] + nDx[0], // U
    imgSrc->widthStep*nDy[1] + nDx[1], // R
    imgSrc->widthStep*nDy[2] + nDx[2], // D
    imgSrc->widthStep*nDy[3] + nDx[3]  // L
  };
  
  int nSeeds=0;
  uchar * pDataSrc = (unsigned char*)imgSrc->imageData;

  while (nStart<nEnd && nEnd<m_imgSize)
  {
    nCurrX = pts[nStart].x;
	nCurrY = pts[nStart].y;
	idxCurr = -1; 

    // Search the 4 connected point of current seed
    for (k=0; k<4; k++) 
    {    
      xx = nCurrX + nDx[k];
      yy = nCurrY + nDy[k];
      if (xx<xMin || xx>xMax || yy<yMin || yy>yMax)
      {
        continue; // not in range of ROI
      }
      if (idxCurr<0)
      {
        idxCurr = imgSrc->widthStep * nCurrY + nCurrX;
      }
	  
      if ( (pDataSrc[ idxCurr + offset[k] ] == 0) && 
		   (pDataSrc[idxCurr]>0&&pDataSrc[idxCurr]<=dist) )
	  {
	    pDataSrc[idxCurr+ offset[k]] = pDataSrc[idxCurr]+1;
	    pts[nSeeds++]=cvPoint(xx,yy);

        ++nEnd;  // Stack end point move forward
        if (nEnd>=m_imgSize)
        {
          break;
        }
	  }
    }
    ++nStart;
  }
}

/** 
 * function f = div(nx,ny)
 *   [nxx,junk]=gradient(nx);
 *   [junk,nyy]=gradient(ny);
 *   f=nxx+nyy;
 * end
 *
 * @param dx    in:  x direction values
 * @param dy    in:  y direction values
 * @param dst   out: divergence matrix
 * @param aperture_size in: 
 */
CVAPI(void) cvCalcDivergence(CvMat * dx, CvMat * dy, CvMat * dst)
{
  int nr=dx->rows, nc=dx->cols;
  int type=CV_MAT_TYPE(dx->type);
  assert( (dx->rows==dy->rows) && (dy->rows==dst->rows) );
  assert( (dx->cols==dy->cols) && (dy->cols==dst->cols) );
  CvMat * dxdx=cvCreateMat(nr, nc, type);
  CvMat * dydy=cvCreateMat(nr, nc, type);
  cvSobel(dx, dxdx, 1, 0, 1);
  cvSobel(dy, dydy, 0, 1, 1);
  cvAdd(dxdx, dydy, dst); cvScale(dst, dst, 0.5);
  cvReleaseMat(&dxdx);
  cvReleaseMat(&dydy);
}

/** 
 * function f = Dirac(x, sigma)
 *   f=(1/2/sigma)*(1+cos(pi*x/sigma));
 *   b = (x<=sigma) & (x>=-sigma);
 *   f = f.*b;
 * end
 * 
 * @param src 
 * @param dst 
 * @param sigma 
 */
CVAPI(void) cvCalcDirac(CvMat * src, CvMat * dst, const float sigma)
{
  const int nr = src->rows, nc = src->cols;
  const float hsig = 0.5/sigma;
  const float psig = CV_PI/sigma;
  assert(CV_MAT_TYPE(src->type)==CV_32F);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  float * ss = src->data.fl;
  float * dd = dst->data.fl;
  const int step = src->step/sizeof(float);
  int i,j;
  const float eps = fabs(sigma);

  memset(dd, 0, dst->step*nr);
  for (i = 0; i < nr; i++)
  {
    for (j = 0; j < nc; j++)
    {
      if ( (ss[j]<=eps) && (ss[j]>=-eps) )
        dd[j]=hsig*(1.+cos(psig*ss[j]));
    }
    ss+=step; dd+=step;
  }
}

/** 
 * MATLAB:
 * function H = calc_heaviside(phi, eps)
 *   b = (x<=eps) & (x>=-eps);
 *   f = ((2.0/eps)*phi + 0.5/pi*sin(pi*x/eps))+0.5;
 *   H = f.*b;
 *   phi<-eps=1e-5;
 *   phi>eps=1-1e-5;
 * 
 * @param src in:  phi
 * @param dst out: Heaviside step function: H_eps
 * @param eps in:  epsilon
 * 
 * @return 
 */
// extern void printminmax(CvMat *);
CVAPI(void) cvCalcHeaviside(CvMat * src, CvMat * dst, const float _eps)
{
  const float eps = fabs(_eps);
  assert(src&&dst);
  assert( (CV_MAT_TYPE(src->type)==CV_32F)&&
          (CV_MAT_TYPE(dst->type)==CV_32F) );
  float * ss = src->data.fl;
  float * dd = dst->data.fl;
  const int nr = src->rows, nc = src->cols;
  const int step = src->step/sizeof(float);
  int i, j;
  // const float lowerval=FLT_EPSILON;
  // const float upperval=1.-FLT_EPSILON;
  const float lowerval=1E-5;
  const float upperval=1.-lowerval;
  const float hieps = 0.5f/eps;
  const float hpi = 0.5f/CV_PI;
  const float pideps = CV_PI/eps;
  for (i = 0; i < nr; i++)
  {
    for (j = 0; j < nc; j++)
    {
      if (ss[j]>eps)
        dd[j] = upperval;
      else if (ss[j]< -eps)
        dd[j] = lowerval;
      else 
        dd[j] = MAX(lowerval,MIN(upperval,
                                 hieps*ss[j]+hpi*sin(pideps*ss[j])+0.5f));
      assert(dd[j]>0||dd[j]<1.001);
    }
    ss+=step; dd+=step;
  }
  // printminmax(dst);
}

/** 
 * Make a function satisfy Neumann boundary condition
 *
 * function g = NeumannBoundCond(f)
 *   [nrow,ncol] = size(f);
 *   g = f;
 *   g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]); % corner elements
 *   g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);     % upper&lower bound
 *   g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);     % left&right bound
 * 
 * @param src in:  f
 * @param dst out: g
 */
CVAPI(void) cvNeumannBoundCond(CvMat * src)
{
  const int nr = src->rows;
  const int nc = src->cols;
  int type = CV_MAT_TYPE(src->type);
  int i;

  assert( type==CV_32F );

  if (type==CV_32F)
  {
    // elements on the corner
    CV_MAT_ELEM(*src, float, 0, 0)   =CV_MAT_ELEM(*src, float, 2, 2);
    CV_MAT_ELEM(*src, float, nr-1, 0)=CV_MAT_ELEM(*src, float, nr-3, 2);
    CV_MAT_ELEM(*src, float, 0, nc-1)=CV_MAT_ELEM(*src, float, 2, nc-3);
    CV_MAT_ELEM(*src, float, nr-1, nc-1)=
        CV_MAT_ELEM(*src, float, nr-3, nc-3);

    // elements on upper and lower bound
    for (i = 1; i < nr-2; i++)
    {
      CV_MAT_ELEM(*src, float, i, 0)=CV_MAT_ELEM(*src, float, i, 2);
      CV_MAT_ELEM(*src, float, i, nc-1)=CV_MAT_ELEM(*src, float, i, nc-3);
    }

    // elements on left and right bound
    for (i = 1; i < nc-2; i++)
    {
      CV_MAT_ELEM(*src, float, 0, i)=CV_MAT_ELEM(*src, float, 2, i);
      CV_MAT_ELEM(*src, float, nr-1, i)=CV_MAT_ELEM(*src, float, nr-3, i);
    }
  }else{
    assert(false);
  }
}

CVAPI(CvLevelSetTracker*) cvCreateLevelSetTracker(
    const CvSize imsize,
    const int type)
{
  CvLevelSetTracker * lstracker = new CvLevelSetTracker;
  memset(lstracker, 0, sizeof(CvLevelSetTracker));
  
  lstracker->type = type;
  lstracker->imsize = imsize;
  int nr=imsize.height,nc=imsize.width;
  
  lstracker->phi        = cvCreateMat(nr, nc, type);
  lstracker->g          = cvCreateMat(nr, nc, type);

  lstracker->bw         = cvCreateMat(nr, nc, CV_8U);
  lstracker->vx         = cvCreateMat(nr, nc, type);
  lstracker->vy         = cvCreateMat(nr, nc, type);
  lstracker->dx         = cvCreateMat(nr, nc, type);
  lstracker->dy         = cvCreateMat(nr, nc, type);
  lstracker->mag        = cvCreateMat(nr, nc, type);
  lstracker->Nx         = cvCreateMat(nr, nc, type);
  lstracker->Ny         = cvCreateMat(nr, nc, type);
  lstracker->dxdx       = cvCreateMat(nr, nc, type);
  lstracker->dydy       = cvCreateMat(nr, nc, type);
  lstracker->curvature  = cvCreateMat(nr, nc, type);
  lstracker->del2       = cvCreateMat(nr, nc, type);
  lstracker->dirac      = cvCreateMat(nr, nc, type);

  lstracker->distRegTerm = cvCreateMat(nr, nc, type);
  lstracker->edgeTerm   = cvCreateMat(nr, nc, type);
  lstracker->areaTerm   = cvCreateMat(nr, nc, type);

  return lstracker;
}

CVAPI(void) cvLevelSetPrepare(CvLevelSetTracker * lstracker,
                              const CvArr * _img,
                              CvRect * rois, int nroi, int ksize)
{
  IplImage img_stub;
  IplImage * img = cvGetImage(_img, &img_stub);
  assert(img->nChannels==1);                    // assume grayscale input
  CvSize imsize=cvGetSize(img);
  int nr=imsize.height,nc=imsize.width,type=lstracker->type;
  CvMat * img_smooth=cvCreateMat(nr, nc, CV_8U);
  int i,j; float aa,bb;
  CvMat * subphi, phi_stub;
  
  if ( (lstracker->imsize.height!=imsize.height) ||
       (lstracker->imsize.width!=imsize.width) )
  {
    cvReleaseMat(&lstracker->phi);
    cvReleaseMat(&lstracker->g);
    cvReleaseMat(&lstracker->bw);
    cvReleaseMat(&lstracker->vx);
    cvReleaseMat(&lstracker->vy);
    cvReleaseMat(&lstracker->dx);
    cvReleaseMat(&lstracker->dy);
    cvReleaseMat(&lstracker->mag);
    cvReleaseMat(&lstracker->Nx);
    cvReleaseMat(&lstracker->Ny);
    cvReleaseMat(&lstracker->dxdx);
    cvReleaseMat(&lstracker->dydy);
    cvReleaseMat(&lstracker->curvature);
    cvReleaseMat(&lstracker->del2);
    cvReleaseMat(&lstracker->dirac);
    cvReleaseMat(&lstracker->distRegTerm);
    cvReleaseMat(&lstracker->edgeTerm);
    cvReleaseMat(&lstracker->areaTerm);

    lstracker->imsize = imsize;
    lstracker->phi        = cvCreateMat(nr, nc, type);
    lstracker->g          = cvCreateMat(nr, nc, type);
    lstracker->bw         = cvCreateMat(nr, nc, CV_8U);
    lstracker->vx         = cvCreateMat(nr, nc, type);
    lstracker->vy         = cvCreateMat(nr, nc, type);
    lstracker->dx         = cvCreateMat(nr, nc, type);
    lstracker->dy         = cvCreateMat(nr, nc, type);
    lstracker->mag        = cvCreateMat(nr, nc, type);
    lstracker->Nx         = cvCreateMat(nr, nc, type);
    lstracker->Ny         = cvCreateMat(nr, nc, type);
    lstracker->dxdx       = cvCreateMat(nr, nc, type);
    lstracker->dydy       = cvCreateMat(nr, nc, type);
    lstracker->curvature  = cvCreateMat(nr, nc, type);
    lstracker->del2       = cvCreateMat(nr, nc, type);
    lstracker->dirac      = cvCreateMat(nr, nc, type);
    lstracker->distRegTerm = cvCreateMat(nr, nc, type);
    lstracker->edgeTerm   = cvCreateMat(nr, nc, type);
    lstracker->areaTerm   = cvCreateMat(nr, nc, type);
  }

  cvSmooth(img, img_smooth, CV_GAUSSIAN, ksize, ksize, 1.5);
  cvSobel(img_smooth, lstracker->dx, 1, 0, 1);
  cvSobel(img_smooth, lstracker->dy, 0, 1, 1);

  {
    float * xx = lstracker->dx->data.fl;
    float * yy = lstracker->dy->data.fl;
    float * gg = lstracker->g->data.fl;
    int step=lstracker->g->step/sizeof(float);
    for (i = 0; i < nr; i++)
    {
      for (j = 0; j < nc; j++)
      {
        xx[j]*=0.5f; yy[j]*=0.5f;
        gg[j]=1.f/((xx[j]*xx[j])+(yy[j]*yy[j])+1.f);
      }
      xx+=step; yy+=step; gg+=step;
    }
  }

  if (nroi>0)
  {
    cvSet(lstracker->phi, cvScalar(2));
    for (i = 0; i < nroi; i++)
    {
      subphi = cvGetSubRect(lstracker->phi, &phi_stub, rois[i]);
      cvSet(subphi, cvScalar(-2));
    }
  }
  
  cvReleaseMat(&img_smooth);
}

CVAPI(int) cvLevelSetUpdate(CvLevelSetTracker * t,
                            float _dt,
                            float _mu, float _lambda, float _alpha, 
                            int _inner_maxiter, int _outer_maxiter)
{
  int i,j,m,n; float aa,bb;
  int nr=t->imsize.height,nc=t->imsize.width;
  int sum,prevsum=0xffffff;

  t->dt=_dt;                 // timestep
  t->mu=_mu;                 // dist term : 0.2f/_dt;
  t->lambda=_lambda;         // edge term : 5.f;
  t->alpha=_alpha;           // area term 
  t->epsilon=1.5f;           // for Dirac delta function
  t->inner_maxiter=_inner_maxiter;
  t->outer_maxiter=_outer_maxiter;

  float dt = t->dt;
  float mu = t->mu;
  int maxiter = t->inner_maxiter;
  float lambda = t->lambda;
  float alpha = t->alpha;
  float epsilon = t->epsilon;
  
  CvMat * bw = t->bw;
  CvMat * vx = t->vx;
  CvMat * vy = t->vy;
  CvMat * dx = t->dx;
  CvMat * dy = t->dy;
  CvMat * dxdx = t->dxdx;
  CvMat * dydy = t->dydy;
  CvMat * mag = t->mag;
  CvMat * Nx = t->Nx;
  CvMat * Ny = t->Ny;
  CvMat * curvature = t->curvature; 
  CvMat * del2 = t->del2;
  CvMat * dirac = t->dirac;
  CvMat * distTerm = t->distRegTerm;
  CvMat * edgeTerm = t->edgeTerm;
  CvMat * areaTerm = t->areaTerm;

  CvMat * phi = t->phi;
  CvMat * g = t->g;

  for (m = 0; m < t->outer_maxiter; m++)
  {
    cvSobel(g, vx, 1, 0, 1); cvScale(vx, vx, 0.5);
    cvSobel(g, vy, 0, 1, 1); cvScale(vy, vy, 0.5);

    for (i = 0; i < maxiter; i++)
    {
      cvNeumannBoundCond(phi);

      cvSobel(phi, dx, 1, 0, 1); 
      cvSobel(phi, dy, 0, 1, 1);

      int step = phi->step/sizeof(float);
      {
        float * xx = dx->data.fl;
        float * yy = dy->data.fl;
        float * mptr = mag->data.fl;
        for (i = 0; i < nr; i++)
        {
          for (j = 0; j < nc; j++)
          {
            xx[j] = xx[j]*0.5f; yy[j]=yy[j]*0.5f;
            mptr[j] = sqrt((xx[j]*xx[j])+(yy[j]*yy[j]))+1E-4;
          }
          xx+=step; yy+=step; mptr+=step;
        }
      }

      cvDiv(dx, mag, Nx);
      cvDiv(dy, mag, Ny);

      cvSobel(Nx, dxdx, 1, 0, 1);
      cvSobel(Ny, dydy, 0, 1, 1);
      cvAdd(dxdx, dydy, curvature);
      cvScale(curvature, curvature, 0.5);

      // distTerm = 4*del2(phi)-curvature;
      cvLaplace(phi, del2, 1);
      cvSub(del2, curvature, distTerm);

      // diracPhi=Dirac(phi,epsilon);
      cvCalcDirac(phi, dirac, epsilon);
      // areaTerm=diracPhi.*g;
      cvMul(dirac, g, areaTerm);
      // edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*curvature;
      {
        cvMul(vx, Nx, Nx); cvMul(vy, Ny, Ny); cvAdd(Nx, Ny, Nx);
        cvMul(dirac, Nx, Nx);
        cvMul(areaTerm, curvature, curvature);
        cvAdd(Nx, curvature, edgeTerm);
      }

      {
        float tm=mu*dt; float tl=lambda*dt; float ta = alpha*dt;
        float * dterm = distTerm->data.fl;
        float * eterm = edgeTerm->data.fl;
        float * aterm = areaTerm->data.fl;
        float * pterm = phi->data.fl;

        for (i = 0; i < nr; i++)
        {
          for (j = 0; j < nc; j++)
          {
            pterm[j]=pterm[j]+tm*dterm[j]+tl*eterm[j]+ta*aterm[j];
          }
          dterm+=step; eterm+=step; aterm+=step; pterm+=step;
        }
      }
    }

    cvCmpS(phi, 0, bw, CV_CMP_LT);
    cvSet(bw, cvScalar(1), bw);
    sum = cvSum(bw).val[0];
    // if (fabs(prevsum-sum)<3) {break;}
    if (prevsum==sum) {break;}
    // else{fprintf(stderr, "%d,", prevsum-sum);}
    prevsum = sum;
  }

  return m;
}

CVAPI(void) cvReleaseLevelSetTracker(CvLevelSetTracker ** lstracker)
{
  CvLevelSetTracker * t = *lstracker;
  CV_FUNCNAME("cvReleaseLevelSetTracker");
  __BEGIN__;
  CV_CALL( cvReleaseMat(&t->phi) );
  CV_CALL( cvReleaseMat(&t->g) );
  CV_CALL( cvReleaseMat(&t->bw) );
  CV_CALL( cvReleaseMat(&t->vx) );
  CV_CALL( cvReleaseMat(&t->vy) );
  CV_CALL( cvReleaseMat(&t->dx) );
  CV_CALL( cvReleaseMat(&t->dy) );
  CV_CALL( cvReleaseMat(&t->mag) );
  CV_CALL( cvReleaseMat(&t->Nx) );
  CV_CALL( cvReleaseMat(&t->Ny) );
  CV_CALL( cvReleaseMat(&t->dxdx) );
  CV_CALL( cvReleaseMat(&t->dydy) );
  CV_CALL( cvReleaseMat(&t->curvature) );
  CV_CALL( cvReleaseMat(&t->del2) );
  CV_CALL( cvReleaseMat(&t->dirac) );
  CV_CALL( cvReleaseMat(&t->distRegTerm) );
  CV_CALL( cvReleaseMat(&t->edgeTerm) );
  CV_CALL( cvReleaseMat(&t->areaTerm) );
  memset(t, 0, sizeof(CvLevelSetTracker));
  delete t;
  __END__;
}
