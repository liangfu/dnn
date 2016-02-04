/**
 * @file   main24_levelset.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Tue Jan 15 10:35:56 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"
#include <string.h>

class CvLevelSetTracking
{
  // CvMat * phi;        // level-set func to be updated
  // CvMat * g;          // edge indicator func
  
  float lambda;       // weighted length term
  float mu;           // weighted distance term
  float alpha;        // weighted area term
  float epsilon;      // width of Dirac Delta function
  float dt;           // time step
  int maxiter;      // number of iterations

 public:
  CvLevelSetTracking()
#define use_data0 1
  {
    lambda=5.f; epsilon=1.5f; maxiter=5;
#if use_data0
    dt=5.f; alpha=1.5f;
#else
    dt=1.f; alpha=-3.f;
#endif
    mu=0.2f/dt;
  }

  ~CvLevelSetTracking()
  {}

  void evolution(CvMat * phi, CvMat * g);
};
  
int main(int argc, char * argv[])
{
  cvNamedWindow("Test");
  // IplImage * img = cvLoadImage("../data/twocells.bmp", 0);
#if use_data0
  IplImage * img = cvLoadImage(argc>1?argv[1]:"../data/twocells.bmp", 0);
#else
  IplImage * img = cvLoadImage(argc>1?argv[1]:"../data/gourd.bmp", 0);
#endif
  const CvSize imsize = cvGetSize(img);
  const int nr = imsize.height;
  const int nc = imsize.width;
  float aa,bb; int i,j;

  const float sigma = 1.5f;
  CvMat * img_smooth = cvCreateMat(nr, nc, CV_8U);
  cvSmooth(img, img_smooth, CV_GAUSSIAN, 15, 15, sigma);

  CvMat * dx = cvCreateMat(nr, nc, CV_32F);
  CvMat * dy = cvCreateMat(nr, nc, CV_32F);
  CvMat * mag = cvCreateMat(nr, nc, CV_32F);
  CvMat * g = cvCreateMat(nr, nc, CV_32F);      // edge indicator func
  cvSobel(img_smooth, dx, 1, 0, 1); // cvScale(dx, dx, 0.5);
  cvSobel(img_smooth, dy, 0, 1, 1); // cvScale(dy, dy, 0.5);
  // cvCartToPolar(dx, dy, mag); // mag = sqrt(dx.^2+dy.^2);
  // cvAddS(mag, cvScalar(1), mag);
  // cvDiv(NULL, mag, g);
  for (i = 0; i < nr; i++)
    for (j = 0; j < nc; j++)
    {
      aa = CV_MAT_ELEM(*dx, float, i, j)*0.5f;
      bb = CV_MAT_ELEM(*dy, float, i, j)*0.5f;
      CV_MAT_ELEM(*g, float, i, j) = 1.f/((aa*aa)+(bb*bb)+1.f);
    }

  CvMat * phi = cvCreateMat(nr, nc, CV_32F);
  cvSet(phi, cvScalar(2));
  CvMat * subphi, phi_stub;
#if use_data0
  subphi = cvGetSubRect(phi, &phi_stub, cvRect(9,9,65,45));
#else
  subphi = cvGetSubRect(phi, &phi_stub, cvRect(20,25,5,10));
  cvSet(subphi, cvScalar(-2));
  subphi = cvGetSubRect(phi, &phi_stub, cvRect(40,25,10,10));
#endif
  cvSet(subphi, cvScalar(-2));
  
  // cvShowImageEx("Test", phi); CV_WAIT();
  // cvShowImageEx("Test", g); CV_WAIT();

  CvLevelSetTracking tracker;
  CvMat * bw = cvCreateMat(nr, nc, CV_8U); int sum,prevsum=0xffffff;
// CV_TIMER_START();
  for (i=0;i<100;i++)
  {
    fprintf(stderr, "iter[%d]: ", i);
    tracker.evolution(phi, g);
    if (1)
    {
      cvCmpS(phi, 0, bw, CV_CMP_LT);
      cvShowImage("Test", bw); CV_WAIT();
      //cvShowImageEx("Test", phi); CV_WAIT();
      cvSet(bw, cvScalar(1), bw);
      sum=cvSum(bw).val[0];
      fprintf(stderr, "%d", prevsum-sum );
      if ( abs(prevsum-sum)==0 ) {break;} 
      prevsum=sum;
    }
    fprintf(stderr, "\n");
  }
// CV_TIMER_SHOW();
  cvReleaseMat(&bw);
  
  cvReleaseMat(&img_smooth);
  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&mag);
  cvReleaseMat(&g);

  cvDestroyAllWindows();
  
  return 0;
}

void CvLevelSetTracking::evolution(CvMat * phi, CvMat * g)
{
  int nr=phi->rows, nc=phi->cols;
  int type=CV_MAT_TYPE(phi->type);

  CvMat * vx = cvCreateMat(nr, nc, CV_32F);
  CvMat * vy = cvCreateMat(nr, nc, CV_32F);
  CvMat * dx = cvCreateMat(nr, nc, CV_32F);
  CvMat * dy = cvCreateMat(nr, nc, CV_32F);
  CvMat * mag = cvCreateMat(nr, nc, CV_32F);
  CvMat * Nx = cvCreateMat(nr, nc, CV_32F);
  CvMat * Ny = cvCreateMat(nr, nc, CV_32F);
  CvMat * curvature = cvCreateMat(nr, nc, CV_32F);
  CvMat * del2 = cvCreateMat(nr, nc, CV_32F);
  CvMat * dirac = cvCreateMat(nr, nc, CV_32F);

  CvMat * distRegTerm = cvCreateMat(nr, nc, CV_32F);
  CvMat * edgeTerm = cvCreateMat(nr, nc, CV_32F);
  CvMat * areaTerm = cvCreateMat(nr, nc, CV_32F);

  CvMat * bw = cvCreateMat(nr, nc, CV_8U);
  int i,j; float aa, bb;

  cvSobel(g, vx, 1, 0, 1); cvScale(vx, vx, 0.5);
  cvSobel(g, vy, 0, 1, 1); cvScale(vy, vy, 0.5);

  for (i = 0; i < maxiter; i++)
  {
    cvNeumannBoundCond(phi);

    cvSobel(phi, dx, 1, 0, 1); 
    cvSobel(phi, dy, 0, 1, 1);

    for (i = 0; i < nr; i++)
      for (j = 0; j < nc; j++)
      {
        CV_MAT_ELEM(*dx, float, i, j) = CV_MAT_ELEM(*dx, float, i, j)*0.5f;
        CV_MAT_ELEM(*dy, float, i, j) = CV_MAT_ELEM(*dy, float, i, j)*0.5f;
      }
    for (i = 0; i < nr; i++)
      for (j = 0; j < nc; j++)
      {
        aa = CV_MAT_ELEM(*dx, float, i, j);
        bb = CV_MAT_ELEM(*dy, float, i, j);
        CV_MAT_ELEM(*mag, float, i, j)=sqrt((aa*aa)+(bb*bb))+1E-4;
      }
    
    cvDiv(dx, mag, Nx);
    cvDiv(dy, mag, Ny);
    cvCalcDivergence(Nx, Ny, curvature); // curvature=div(Nx,Ny);

    // distRegTerm = 4*del2(phi)-curvature;
    cvLaplace(phi, del2, 1);
    cvSub(del2, curvature, distRegTerm);

    // diracPhi=Dirac(phi,epsilon);
    cvCalcDirac(phi, dirac, epsilon);
    // areaTerm=diracPhi.*g;
    cvMul(dirac, g, areaTerm);
    // edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*curvature;
    {
      cvMul(vx, Nx, Nx); cvMul(vy, Ny, Ny); cvAdd(Nx, Ny, Nx);
      cvMul(dirac, Nx, Nx);                     // destroy "Nx", "Ny"

      cvMul(dirac, curvature, curvature);
      cvMul(g, curvature, curvature);
      cvAdd(Nx, curvature, edgeTerm);
    }

    // phi=phi + timestep*(mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm);
    {
      cvScale(distRegTerm, distRegTerm, mu*dt);
      cvScale(edgeTerm, edgeTerm, lambda*dt);
      cvScale(areaTerm, areaTerm, alpha*dt);
// cvShowImageEx("Test", distRegTerm); CV_WAIT();
// cvShowImageEx("Test", edgeTerm); CV_WAIT();
// cvShowImageEx("Test", areaTerm); CV_WAIT();
      cvAdd(phi, distRegTerm, phi);
      cvAdd(phi, edgeTerm, phi);
      cvAdd(phi, areaTerm, phi);
    }
  }
  
  cvReleaseMat(&vx);
  cvReleaseMat(&vy);
  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&mag);
  cvReleaseMat(&Nx);
  cvReleaseMat(&Ny);
  cvReleaseMat(&curvature);
  cvReleaseMat(&dirac);
  cvReleaseMat(&del2);
  cvReleaseMat(&distRegTerm);
  cvReleaseMat(&edgeTerm);
  cvReleaseMat(&areaTerm);
}

