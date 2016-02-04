/**
 * @file   cvpfilter.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jan  7 17:50:21 2013
 * 
 * @brief  
 * 
 */

#include "cvparticleobserve.h"
#include "cvimgwarp.h"
#include "cvhog.h"
#include "cvtimer.h"

void icvCalcWarpHistYUV(CvMat * imgY, CvMat * imgU, CvMat * imgV,
                        CvMat * warp_p, CvSize roisize,
                        CvMatND * m_currhist, int bound);
void icvCalcHist(
    CvMat * imgYpatch, CvMat * imgUpatch, CvMat * imgVpatch,
    CvMatND * reffhist, CvMat * submask=0);

void CvParticleObserve::initialize(CvRect roi)
{
  CvSize imsize = cvGetSize(m_tracker.m_imgY);

  float lin = imsize.width/160.;
  int bound = floor(5*lin);
  float warp_p_data[3] = {
    float(roi.height)/float(hogsizes[0]*6), roi.x, roi.y
  };
  CvMat warp_p = cvMat(3,1,CV_32F,warp_p_data);

  // initialize full map of gradient and angular values upon the image
  {
  if (!magni_full){
    magni_full=cvCreateMat(imsize.height,imsize.width,CV_32F); }
  if (!angle_full){
    angle_full=cvCreateMat(imsize.height,imsize.width,CV_32F); }
  }
  cvSet(magni_full, cvScalar(-1));

  {
    // calculate histogram of foreground 
    icvCalcWarpHistYUV(imgY, imgU, imgV, &warp_p,
                       cvSize(roi.width,roi.height), m_reffhist, bound);

    // extract texture descriptor
    warp_p_data[1]+=bound;
    warp_p_data[2]+=bound;
    icvCalcWarpHOG(imgY,&warp_p,ref0hog,6,2,
                   m_tracker.dx,m_tracker.dy,magni_full,angle_full);
    cvScale(ref0hog,ref0hog,1./cvSum(ref0hog).val[0]);
    
    // int hogszprod=hogsizes[0]*hogsizes[1]*hogsizes[2];
    // for (int i=0;i<hogszprod;i++){
    //   fprintf(stderr, "%ff,", ref0hog->data.fl[i]);
    // }

    // float meanhog0[]={
    // };
    // memcpy(ref0hog->data.fl,meanhog0,sizeof(float)*378);

    float meanhog1[378]={
0,
    };
    memcpy(ref1hog->data.fl,meanhog1,sizeof(float)*378);
    
// CV_SHOW(angle_full);
// CV_SHOW(magni_full);
// icvShowHOG(ref0hog); CV_WAIT();
  }
  
  m_initialized=1;
}

int CvParticleObserve::measure(CvParticle * particle)
{
  assert(initialized());
  int i,j;
  int cellsize=6;
  float lin = cvGetSize(imgY).width/160.;
  int bound = 5*lin;

  cvSet(magni_full, cvScalar(-1));

  CvSize roisize = 
      cvSize(ref0hog->dim[1].size*cellsize, ref0hog->dim[0].size*cellsize);

  static int statusbuff[1000]={0,}; 
  static float hogfeature_data[378];
  CvMat hogfeature = cvMat(1,378,CV_32F,hogfeature_data);
  static float result_data[2];
  CvMat result = cvMat(1,2,CV_32F,result_data);

  for (i = 0; i < particle->num_particles; i++)
  {
    // crop image roi
    CvMat warp_p;
    {
      int xloc     = CV_MAT_ELEM(*particle->particles,float,0,i);
      int yloc     = CV_MAT_ELEM(*particle->particles,float,1,i);
      float height = CV_MAT_ELEM(*particle->particles,float,3,i);
      float angle  = CV_MAT_ELEM(*particle->particles,float,4,i);
      float warp_p_data[3] = {
        float(hogsizes[0]*6)/height, //*cos(angle/180.*CV_PI),
        xloc-bound,yloc-bound
      };
      warp_p = cvMat(3,1,CV_32F,warp_p_data);
    }
    
// CV_TIMER_START();
    {
      // icvCalcHist(imgYpatch,imgUpatch,imgVpatch,m_currhist,fgmask);
      icvCalcWarpHistYUV(imgY,imgU,imgV,&warp_p,
                         cvSize(roisize.width *warp_p.data.fl[0],
                                roisize.height*warp_p.data.fl[0]),
                         m_currhist,bound);
// CV_SHOW(magni_full);
// CV_TIMER_START();
      // icvCalcWarpHOG(imgY,&warp_p,currhog,6,cvSize(cbound,rbound),
      //                m_tracker.dx,m_tracker.dy,magni_full,angle_full);
      warp_p.data.fl[1]+=bound;
      warp_p.data.fl[2]+=bound;
      icvCalcWarpHOG(imgY,&warp_p,currhog,6,2,
                     m_tracker.dx,m_tracker.dy,magni_full,angle_full);
{
memcpy(hogfeature_data,currhog->data.fl,sizeof(float)*378);
m_lda4hog.predict_withprior(&hogfeature,&result);
// fprintf(stderr, "result: %.2f\n", result_data[1]);
statusbuff[i] = result_data[1]>0.5;
}
      cvScale(currhog,currhog,1./cvSum(currhog).val[0]);
      
// CV_TIMER_SHOW();
// cvShowImage("Test",imgYpatch);cvWaitKey();
// CV_SHOW(magni_full);
// CV_SHOW(angle_full);
// cvPrintMinMax(angle_full);
// CV_SHOW(imgYpatch);
// icvShowHOG(currhog); CV_WAIT();
    }

    double totalweights = 0;

    // get histogram and compare 
    {
      if ( (!m_reffhist) || (!m_refbhist) ) {
        fprintf(stderr, "ERROR: previous histogram not initialized!\n");
        return 0;
      }

      // BHATTACHARYYA histogram normalization
      {
        float * curptr = m_currhist->data.fl;
        float * refptr = m_reffhist->data.fl;
        float * ref0hogptr = ref0hog->data.fl;
        float * ref1hogptr = ref1hog->data.fl;
        float * currhogptr = currhog->data.fl;
        int hogsizeprod =
            ref0hog->dim[0].size*ref0hog->dim[1].size*ref0hog->dim[2].size;
        double sum=0,sum0=0,sum1=0;
        double hogweight = .5;
        double yuvweight = 1.-hogweight;
        for (j=0;j<nbins*nbins*nbins;j++){
          sum += sqrt(curptr[j]*refptr[j]);
        }
        sum *= yuvweight;
        for (j=0;j<hogsizeprod;j++){
          sum0 += sqrt(currhogptr[j]*ref0hogptr[j]);
          sum1 += sqrt(currhogptr[j]*ref1hogptr[j]);
        }
        sum0 *= hogweight;
        sum1 *= hogweight;
        sum += MAX(sum0,sum1);

        //statusbuff[i] = sum0>sum1?0:1;

        //fprintf(stderr, "%.2f,", sum);
        totalweights = -log(sqrt(1-sum));
      }
    }
// CV_TIMER_SHOW();

    particle->weights->data.db[i] = totalweights;
  }

  // learn with least weighted particles
  static int weightsbuffiter=0;
  static float weightsBuffer[30]={0,};
  {
    int maxloc =-1,minloc=-1;
    double maxval=-0xffff,minval=0xffff;
    {
      int ncols=particle->weights->cols;
      double * wptr = particle->weights->data.db;
      for (i=0;i<ncols;i++){
        if (maxval<wptr[i]){maxval=wptr[i];maxloc=i;}
        if (minval>wptr[i]){minval=wptr[i];minloc=i;}
      }
    }

    m_status=statusbuff[maxloc];
    
#if 1
    {
    weightsBuffer[weightsbuffiter++%30]=maxval;
    CvMat meanWeights=cvMat(30,1,CV_32F,weightsBuffer);
    float meanWeightsVal=cvAvg(&meanWeights).val[0];
    // if (meanWeightsVal<0.7){m_initialized=0;}
    }
#endif
  }

  return 1;
}

void icvCalcHist(
    CvMat * imgYpatch, CvMat * imgUpatch, CvMat * imgVpatch,
    CvMatND * reffhist, CvMat * submask)
{
  cvZero(reffhist); 
  uchar * yptr = imgYpatch->data.ptr;
  uchar * uptr = imgUpatch->data.ptr;
  uchar * vptr=imgVpatch->data.ptr;
  int ystep=reffhist->dim[0].step/sizeof(float),
      ustep=reffhist->dim[1].step/sizeof(float),
      vstep=reffhist->dim[2].step/sizeof(float);
  float * reffptr=reffhist->data.fl;
  int j,k,step=imgYpatch->step/sizeof(uchar);
  int nr=imgYpatch->rows,nc=imgYpatch->cols;
  assert(reffhist->dim[0].size==8);
  assert(reffhist->dim[1].size==8);
  assert(reffhist->dim[2].size==8);
  // int yshift=log2(256/reffhist->dim[0].size);
  // int ushift=log2(256/reffhist->dim[1].size);
  // int vshift=log2(256/reffhist->dim[2].size);
  if (!submask){
    for (j=0;j<nr;j++)
    {
      for (k=0;k<nc;k++)
      {
        (reffptr+ystep*(yptr[k]>>5)+ustep*(uptr[k]>>5))[vptr[k]>>5] += 1;
      }
      yptr+=step;uptr+=step;vptr+=step;
    }
  }else{
    assert( (nr==submask->rows)&&(nc==submask->cols) );
    uchar * maskptr=submask->data.ptr;
    int maskstep=submask->step/sizeof(uchar);
    for (j=0;j<nr;j++)
    {
      for (k=0;k<nc;k++)
      {
        if (maskptr[k])
        {
          (reffptr+ystep*(yptr[k]>>5)+ustep*(uptr[k]>>5))[vptr[k]>>5] += 1;
        }
      }
      yptr+=step;uptr+=step;vptr+=step;maskptr+=maskstep;
    }
  }
  // normalize histogram 
  cvScale(reffhist, reffhist, 1./cvSum(reffhist).val[0]);
}

void icvCalcWarpHistYUV(CvMat * imgY, CvMat * imgU, CvMat * imgV,
                        CvMat * warp_p, CvSize roisize, 
                        CvMatND * hist, int bound)
{
  int i,j,k;
  int nr=roisize.height-bound*2,nc=roisize.width;
  float * pptr = warp_p->data.fl;
  float * reffptr=hist->data.fl;
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  if ((warp_p->cols==1)&&(warp_p->rows==4)){
    cp0=pptr[0]; sp1=-pptr[1]; sp3=pptr[1]; cp4=pptr[0];
  }else if ((warp_p->cols==1)&&(warp_p->rows==3)){
    cp0=pptr[0]; sp1=0; sp3=0; cp4=pptr[0];
  }else{
    fprintf(stderr, "WARNING: unknown warp parameter setting !\n");
    assert(false);
  }
  int txid=warp_p->rows-2,tyid=warp_p->rows-1;
  int ystep=hist->dim[0].step/sizeof(float);
  int ustep=hist->dim[1].step/sizeof(float);
  int vstep=hist->dim[2].step/sizeof(float);
  int step=imgY->step/sizeof(uchar);
  assert(CV_MAT_TYPE(imgY->type)==CV_8U);
  assert(CV_MAT_TYPE(imgU->type)==CV_8U);
  assert(CV_MAT_TYPE(imgV->type)==CV_8U);
  assert(step==imgU->step/sizeof(uchar));
  assert(step==imgV->step/sizeof(uchar));
  uchar * yptr = imgY->data.ptr+step*int(pptr[tyid]+bound);
  uchar * uptr = imgU->data.ptr+step*int(pptr[tyid]+bound);
  uchar * vptr = imgV->data.ptr+step*int(pptr[tyid]+bound);
  // uchar * yptr = imgY->data.ptr+step*int(pptr[tyid]);
  // uchar * uptr = imgU->data.ptr+step*int(pptr[tyid]);
  // uchar * vptr = imgV->data.ptr+step*int(pptr[tyid]);

  float txval=pptr[txid];
  cvZero(hist);

// CvMat * img = cvCreateMat(nr,nc,CV_8U); cvZero(img);
  for (i=0;i<nr;i++){
  for (j=bound;j<nc-bound;j++){
    k = (j+txval); 
    (reffptr+ystep*(yptr[k]>>5)+ustep*(uptr[k]>>5))[vptr[k]>>5] += 1;
// CV_MAT_ELEM(*img,uchar,i,j)=yptr[k];
  }
  yptr+=step;uptr+=step;vptr+=step;
  }
  cvScale(hist, hist, 1./cvSum(hist).val[0]);
// CV_SHOW(img);
// cvReleaseMat(&img);
}
