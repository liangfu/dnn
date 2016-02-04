/**
 * @file   cvhog.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Jun  4 17:53:12 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvhog.h"
#include "cvtimer.h"

extern float gauss12x12[144];
extern float gauss18x18[324];

void icvCalcHOG(CvMat * imgYpatch, CvMatND * hog, int ncells, int ngrids)
{
  if (ncells<0){ncells=6;}
  if (ngrids<0){ngrids=1;}
  int nr=imgYpatch->rows,nc=imgYpatch->cols;
  // int rbound = (nr-cvFloor(nr/6)*6)/2; assert(rbound==1);
  // int cbound = (nc-cvFloor(nc/6)*6)/2; assert(cbound==1);
  int rbound = (nr-(ngrids-1)*ncells-hog->dim[0].size*ncells)/2;
  int cbound = (nc-(ngrids-1)*ncells-hog->dim[1].size*ncells)/2;
  assert(rbound==1); assert(cbound==1);
  CvMat * dx = cvCreateMat(nr,nc,CV_16S);
  CvMat * dy = cvCreateMat(nr,nc,CV_16S);
  CvMat * magni = cvCreateMat(nr,nc,CV_32F);
  CvMat * angle = cvCreateMat(nr,nc,CV_32S);
  assert(hog->dim[0].size*6==nr-rbound*2-(ngrids-1)*ncells);
  assert(hog->dim[1].size*6==nc-cbound*2-(ngrids-1)*ncells);
  assert(hog->dim[2].size==9);
  // assert(CV_MAT_TYPE(magni->type)==CV_MAT_TYPE(angle->type));
  assert(magni->rows==angle->rows);
  assert(magni->cols==angle->cols);

  cvSobel(imgYpatch,dx,1,0,1); 
  cvSobel(imgYpatch,dy,0,1,1); 
  cvSet(magni,cvScalar(0));
  cvSet(angle,cvScalar(0));
  {
    int step = dx->step/sizeof(short);
    int magstep = magni->step/sizeof(float);
    int angstep = angle->step/sizeof(int);
    short * dxptr = dx->data.s+rbound*step;
    short * dyptr = dy->data.s+rbound*step;
    float * magptr = magni->data.fl+rbound*magstep;
    int   * angptr = angle->data.i +rbound*angstep;
    int i,j; int tmp;
    for (i=rbound;i<nr-rbound;i++){
    for (j=0;j<nc;j++){
      magptr[j]=sqrt(pow(float(dxptr[j]),2)+pow(float(dyptr[j]),2)+1e-5);
      tmp=floor(atan2(float(dyptr[j]),float(dxptr[j]))/3.15*9.);
      angptr[j]=(tmp>=0)?tmp:-tmp-1;
    }
    dxptr+=step;dyptr+=step;magptr+=magstep;angptr+=angstep;
    }
  }
  // cvPrintMinMax(magni);
  // cvPrintMinMax(angle);
  // CV_SHOW(magni);
  // CV_SHOW(angle);

  int xloc,yloc,m,n,i,j,k,xpos,ypos,bsize=ncells*ngrids,
      hognr=hog->dim[0].size,hognc=hog->dim[1].size,
      txval=cbound,tyval=rbound,ww=nc,hh=nr;
  int hog0step=hog->dim[0].step/sizeof(float),
      hog1step=hog->dim[1].step/sizeof(float),
      magstep=magni->step/sizeof(float),
      angstep=angle->step/sizeof(int);
  double magval,bsum; int angval;
  float * magptr = magni->data.fl+rbound*magstep;
  int   * angptr = angle->data.i +rbound*angstep;
  float * hogptr = hog->data.fl;
  // float * gaussptr = (ngrids==1)?NULL:((ngrids==2)?gauss12x12:gauss18x18);

  cvZero(hog);
  for (m=0;m<hognr;m++){
  yloc = m*ncells+tyval;
  for (n=0;n<hognc;n++){
    xloc = n*ncells+txval;
    if ( (xloc>ww-1)||(xloc<0)||(yloc>hh-1)||(yloc<0) ){
      assert(false);continue;
    }
    hogptr = hog->data.fl+hog0step*m+hog1step*n;
    // for each block
    for (i=0,k=0;i<bsize;i++,k++){
    ypos = i+yloc;
    magptr = magni->data.fl+magstep*ypos;
    angptr = angle->data.i +angstep*ypos;
    for (j=0;j<bsize;j++){
      xpos = j+xloc;
      magval = magptr[xpos];
      angval = angptr[xpos];
      // if (ngrids>1) { hogptr[angval] += magval*gaussptr[k]; }
      if (ngrids==2) { hogptr[angval] += magval*gauss12x12[k]; }
      else if (ngrids==3) { hogptr[angval] += magval*gauss18x18[k]; }
      else if (ngrids==1) { hogptr[angval] += magval; }
      else {assert(false);}
    }
    }

    // normalize block
    bsum=0;
    for (k=0;k<9;k++) { bsum+=hogptr[k]*hogptr[k]; }
    bsum=1./(sqrt(bsum)+.01);
    for (k=0;k<9;k++) { hogptr[k]*=bsum; }
  }    
  }
  // cvScale(hog,hog,1./cvSum(hog).val[0]);
  
  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&magni);
  cvReleaseMat(&angle);
}

int icvCalcHOG_optimized(CvMat * img, CvMatND * hog, int ncells, int ngrids)
{
  // icvCalcHOG(img,hog,ncells,ngrids);
  int rbound=1,cbound=1;
  int i,nr=img->rows,nc=img->cols;
  const int numangles = 9;
  CvMat * magimgs[numangles]={0,};
  CvMat * magints[numangles]={0,};
  for (i=0;i<numangles;i++){
    magimgs[i]=cvCreateMat(nr,nc,CV_32F); cvZero(magimgs[i]);
    magints[i]=cvCreateMat(nr,nc,CV_32F); cvZero(magints[i]);
  }
  CvMat * dx = cvCreateMat(nr,nc,CV_16S);
  CvMat * dy = cvCreateMat(nr,nc,CV_16S);
  CvMat * angle = cvCreateMat(nr,nc,CV_32S);  cvZero(angle);
  CvMat * magni = cvCreateMat(nr,nc,CV_32F);  cvZero(magni);

  cvSobel(img,dx,1,0,1);
  cvSobel(img,dy,0,1,1);

  // calculate angle and magnitude values 
  {
    int step = dx->step/sizeof(short);
    int magstep = magni->step/sizeof(float);
    int angstep = angle->step/sizeof(int);
    short * dxptr = dx->data.s+rbound*step;
    short * dyptr = dy->data.s+rbound*step;
    float * magptr = magni->data.fl+rbound*magstep;
    int   * angptr = angle->data.i +rbound*angstep;
    int i,j; int tmp;
    for (i=rbound;i<nr-rbound;i++){
    for (j=0;j<nc;j++){
      magptr[j]=sqrt(pow(float(dxptr[j]),2)+pow(float(dyptr[j]),2)+1e-5);
      tmp=floor(icvFastAtan2(float(dyptr[j]),float(dxptr[j]))/3.15*9.);
      angptr[j]=(tmp>=0)?tmp:-tmp-1;
    }
    dxptr+=step;dyptr+=step;magptr+=magstep;angptr+=angstep;
    }
  }
  CV_SHOW(angle);
  CV_SHOW(magni);

  // separate magnitude images according to values in `angle` image
  {
    int step = dx->step/sizeof(short);
    int magstep = magni->step/sizeof(float);
    int angstep = angle->step/sizeof(int);
    float * magptr = magni->data.fl+rbound*magstep;
    int   * angptr = angle->data.i +rbound*angstep;
    int i,j; 
    for (i=rbound;i<nr-rbound;i++){
    for (j=0;j<nc;j++){
      (magimgs[angptr[j]]->data.fl+magstep*i)[j]=magptr[j];
    }
    magptr+=magstep;angptr+=angstep;
    }
  }
  for (i=0;i<numangles;i++){ CV_SHOW(magimgs[i]); }

  // calculate integral of separated magnitude images
  for (i=0;i<numangles;i++) {
    cvIntegral(magimgs[i],magints[i]);
  }

  int xloc,yloc,m,n,j,k,xpos,ypos,bsize=ncells*ngrids,
      hognr=hog->dim[0].size,hognc=hog->dim[1].size,
      txval=cbound,tyval=rbound;
  int hog0step=hog->dim[0].step/sizeof(float),
      hog1step=hog->dim[1].step/sizeof(float),
      hogintstep=magints[0]->step/sizeof(float);
  float * hogptr=0;
  int bsum;
  cvZero(hog);
  for (m=0;m<hognr;m++){
  yloc = m*ncells+tyval;
  for (n=0;n<hognc;n++){
    xloc = n*ncells+txval;
    if ( (xloc>nc-1)||(xloc<0)||(yloc>nr-1)||(yloc<0) ){
      assert(false);continue;
    }
    hogptr = hog->data.fl+hog0step*m+hog1step*n;
    // for each block
    for (j=0;j<numangles;j++)
    {
      // hogptr[j]=
      //     (magints[j]->data.fl+hogintstep*(yloc))[xloc]+
      //     (magints[j]->data.fl+hogintstep*(yloc+bsize)[xloc+bsize]-
      //     (magints[j]->data.fl+hogintstep*(yloc)[xloc+bsize]-
      //     (magints[j]->data.fl+hogintstep*(yloc+bsize)[xloc];
    }
    // normalize block
    bsum=0;
    for (k=0;k<9;k++) { bsum+=hogptr[k]*hogptr[k]; }
    bsum=1./(cvSqrt(bsum)+.01);
    for (k=0;k<9;k++) { hogptr[k]*=bsum; }
  }    
  }
  
  for (i=0;i<numangles;i++){
    cvReleaseMat(&magimgs[i]); magimgs[i]=NULL;
    cvReleaseMat(&magints[i]); magints[i]=NULL;
  }
  cvReleaseMat(&dx);
  cvReleaseMat(&dy);
  cvReleaseMat(&angle);
  cvReleaseMat(&magni);

  return 1;
}

/** 
 * efficiently compute HOG from warping of full image
 */
// void icvCalcWarpHOG(CvMat * imgY, CvMat * warp_p, 
//                     CvMatND * hog, int cellsize, CvSize bound,
//                     //int ncells, int ngrids,
//                     CvMat * dx, CvMat * dy,
//                     CvMat * magni_full, CvMat * angle_full)
// {
//   int i,j,m,n;
//   int rbound = bound.height;
//   int cbound = bound.width;
//   int nr=hog->dim[0].size*cellsize+rbound*2;
//   int nc=hog->dim[1].size*cellsize+cbound*2;
//   int hh=imgY->rows,ww=imgY->cols;
//   uchar * imgptr = imgY->data.ptr;
//   float * pptr = warp_p->data.fl;
//   float * hogptr = hog->data.fl;
//   short * dxptr = dx->data.s;
//   short * dyptr = dy->data.s;
//   float * magptr = magni_full->data.fl;
//   float * angptr = angle_full->data.fl;
//   float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
//   // CvPoint pos;
//   int xloc,yloc;
//   int txid=warp_p->rows-2,tyid=warp_p->rows-1;
//   double dxval,dyval,magval,tmpval;int angval;
//   int imgstep = imgY->step/sizeof(uchar);
//   int hog0step = hog->dim[0].step/sizeof(float);
//   int hog1step = hog->dim[1].step/sizeof(float);
//   int dxstep = dx->step/sizeof(short); 
//   int dystep = dy->step/sizeof(short);
//   int magstep = magni_full->step/sizeof(float);
//   int angstep = angle_full->step/sizeof(float);
//   int ridx,cidx;
//   // int blocksize = ncells*ngrids;
//   // same size as input image
//   assert(CV_MAT_TYPE(dx->type)==CV_16S);
//   assert((imgY->rows==magni_full->rows)&&(imgY->cols==magni_full->cols));
//   assert((imgY->rows==angle_full->rows)&&(imgY->cols==angle_full->cols));
//   if ((warp_p->cols==1)&&(warp_p->rows==4)){
//     cp0=pptr[0]; sp1=-pptr[1]; sp3=pptr[1]; cp4=pptr[0];
//   }else if ((warp_p->cols==1)&&(warp_p->rows==3)){
//     cp0=pptr[0]; sp1=0; sp3=0; cp4=pptr[0];
//   }else{
//     fprintf(stderr, "WARNING: unknown warp parameter setting !\n");
//     assert(false);
//   }
//   cvZero(hog);
//   for (i=rbound+cellsize;i<nr-rbound-cellsize;i++){
//   for (j=cbound+cellsize;j<nc-cbound-cellsize;j++){
//     // xloc = cvRound(j*cp0+i*sp1+pptr[txid]); // nearest-neighbor method
//     // yloc = cvRound(j*sp3+i*cp4+pptr[tyid]);
//     xloc = (j+pptr[txid]); // nearest-neighbor method
//     yloc = (i+pptr[tyid]);
//     if ( (xloc>ww-1) || (xloc<0) || (yloc>hh-1) || (yloc<0) ){
//       // do nothing outside of boundary
//     }else{
//       magval = (magptr+magstep*yloc)[xloc];
//       angval = (angptr+angstep*yloc)[xloc];
//       if (magval<0) {
//         dxval = (dxptr+dxstep*yloc)[xloc];
//         dyval = (dyptr+dystep*yloc)[xloc];
//         magval = pow(dxval,2)+pow(dxval,2);
//         tmpval = atan2(dyval,dxval);
//         angval = floor(((tmpval>0)?tmpval:(tmpval+CV_PI))*2.86478);
//         (magptr+magstep*yloc)[xloc]=magval;
//         (angptr+angstep*yloc)[xloc]=angval;
//       }
//       // if (angval>=0)
//       {
//         ridx=(int)(int(i-rbound)/cellsize);
//         cidx=(int)(int(j-cbound)/cellsize);
//         assert(ridx>=0&&ridx<hog->dim[0].size);
//         assert(cidx>=0&&cidx<hog->dim[1].size);
//         (hogptr+hog0step*ridx+hog1step*cidx)[angval] += magval;
//       }
//     }
//   }
//   }
//   cvScale(hog,hog,1./cvSum(hog).val[0]);
// }

/** 
 * efficiently compute HOG from warping of full image
 */
void icvCalcWarpHOG(CvMat * imgY, CvMat * warp_p, 
                    CvMatND * hog, int ncells, int ngrids,
                    CvMat * dx, CvMat * dy,
                    CvMat * magni_full, CvMat * angle_full)
{
  int i,j,k,m,n;
  int nr=hog->dim[0].size*ncells;
  int nc=hog->dim[1].size*ncells;
  int hh=imgY->rows,ww=imgY->cols;
  uchar * imgptr = imgY->data.ptr;
  float * pptr = warp_p->data.fl;
  float * hogptr = hog->data.fl;
  short * dxptr = dx->data.s;
  short * dyptr = dy->data.s;
  float * magptr = magni_full->data.fl;
  float * angptr = angle_full->data.fl;
  float cp0=pptr[0], sp1=-pptr[1], sp3=pptr[1], cp4=pptr[0];
  int xloc,yloc;
  int txid=warp_p->rows-2,tyid=warp_p->rows-1;
  double dxval,dyval,magval,tmpval;int angval;
  int imgstep = imgY->step/sizeof(uchar);
  int hog0step = hog->dim[0].step/sizeof(float);
  int hog1step = hog->dim[1].step/sizeof(float);
  int dxstep = dx->step/sizeof(short); 
  int dystep = dy->step/sizeof(short);
  int magstep = magni_full->step/sizeof(float);
  int angstep = angle_full->step/sizeof(float);
  int ridx,cidx;
  int hognr=hog->dim[0].size;
  int hognc=hog->dim[1].size;
  float * gaussptr = (ngrids==1)?NULL:((ngrids==2)?gauss12x12:gauss18x18);
  assert(ncells==6);

  // same size as input image
  assert(CV_MAT_TYPE(dx->type)==CV_16S);
  assert(CV_MAT_TYPE(dy->type)==CV_16S);
  assert(CV_MAT_TYPE(magni_full->type)==CV_32F);
  assert(CV_MAT_TYPE(angle_full->type)==CV_32F);
  assert((imgY->rows==magni_full->rows)&&(imgY->cols==magni_full->cols));
  assert((imgY->rows==angle_full->rows)&&(imgY->cols==angle_full->cols));

  if ((warp_p->cols==1)&&(warp_p->rows==4)){
    cp0=pptr[0]; sp1=-pptr[1]; sp3=pptr[1]; cp4=pptr[0];
  }else if ((warp_p->cols==1)&&(warp_p->rows==3)){
    cp0=pptr[0]; sp1=0; sp3=0; cp4=pptr[0];
  }else{
    fprintf(stderr, "WARNING: unknown warp parameter setting !\n");
    assert(false);
  }

  cvZero(hog);

  int txval = pptr[txid];
  int tyval = pptr[tyid];
  int bsize = ncells*ngrids;
  int xpos,ypos;double bsum;
  
// CV_TIMER_START();
  m=0;n=0;
  for (m=0;m<hognr;m++){
    // yloc = m*ncells+tyval-(bsize-ncells)*.5;
    yloc = cvRound(float(float(n*ncells)-float(bsize-ncells)*.5)*sp3+
                   float(float(m*ncells)-float(bsize-ncells)*.5)*cp4+
                   float(tyval));
  for (n=0;n<hognc;n++){
    // xloc = n*ncells+txval-(bsize-ncells)*.5;
    xloc = cvRound(float(float(n*ncells)-float(bsize-ncells)*.5)*cp0+
                   float(float(m*ncells)-float(bsize-ncells)*.5)*sp1+
                   float(txval));
    if ( (xloc>ww-bsize*cp0-1)||(xloc<0)||
         (yloc>hh-bsize*cp0-1)||(yloc<0) ){
      continue;
    }
    hogptr = hog->data.fl+hog0step*m+hog1step*n;
    
    // for each block
    j=0;i=0;
    for (i=0,k=0;i<bsize;i++,k++){
    // ypos = i*cp0+yloc;
    ypos = j*sp3+i*cp4+yloc;
    magptr = magni_full->data.fl+magstep*ypos;
    angptr = angle_full->data.fl+angstep*ypos;
    for (j=0;j<bsize;j++){
      // xpos = j*cp0+xloc;
      xpos = j*cp0+i*sp1+xloc;

      magval = magptr[xpos];
      angval = angptr[xpos];
      if (magval<0) {
        dxval = (dxptr+dxstep*ypos)[xpos];
        dyval = (dyptr+dystep*ypos)[xpos];
        magval = pow(dxval,2)+pow(dxval,2);
        tmpval = atan2(dyval,dxval); // *9/CV_PI
        angval = cvFloor(((tmpval>0)?tmpval:tmpval+CV_PI)*2.86478);
        magptr[xpos]=magval;
        angptr[xpos]=angval;
      }

      if (ngrids>1) { hogptr[angval] += magval*gaussptr[k]; }
      else { hogptr[angval] += magval; }
    }
    }

    // normalize block
    bsum=0;
    for (k=0;k<9;k++) { bsum+=hogptr[k]*hogptr[k]; }
    bsum=1./(sqrt(bsum)+.01);
    for (k=0;k<9;k++) { hogptr[k]*=bsum; }
  }    
  }
  // cvScale(hog,hog,1./cvSum(hog).val[0]);
// CV_TIMER_SHOW();
  
  // CV_SHOW(magni_full);
  // CV_SHOW(angle_full);
}

void icvGetGauss(CvMat * gauss, double sigma, int theta);
void icvShowHOG(CvMatND * hog, int cmflag, int scale)
{
  CvMat * disp = cvCreateMat(hog->dim[0].size*6,hog->dim[1].size*6,CV_32F);
  CvMat * gauss = cvCreateMat(6,6,CV_32F);
  cvZero(disp);
  int i,j,k;
  // fprintf(stderr,"%d,%d,%d\n",
  //         hog->dim[0].size,hog->dim[1].size,hog->dim[2].size);
  for (i=0;i<hog->dim[0].size;i++){
  for (j=0;j<hog->dim[1].size;j++){
  for (k=0;k<hog->dim[2].size;k++){
    CvMat subwin_stub;
    CvMat * subwin =
        cvGetSubRect(disp,&subwin_stub,cvRect(j*6,i*6,6,6));
    icvGetGauss(gauss,cvGetReal3D(hog,i,j,k),k);
    cvAdd(subwin,gauss,subwin);
  }
  }    
  }
  // cvShowImageEx("Test", disp, CV_CM_GRAY); cvWaitKey(0);

  //if (MIN(disp->rows,disp->cols)<160)
  if (scale!=1)
  {
    CvMat * disp_resized =
        cvCreateMat(disp->rows*scale,disp->cols*scale,
                    CV_MAT_TYPE(disp->type));
    cvResize(disp,disp_resized);
    cvShowImageEx("Test", disp_resized, cmflag); //cvWaitKey(0);
    cvReleaseMat(&disp_resized);
  }else{
    cvShowImageEx("Test", disp, cmflag); //cvWaitKey(0);
  }
  cvReleaseMat(&disp);
  cvReleaseMat(&gauss);
}

void icvGetGauss(CvMat * gauss, double sigma, int theta)
{
  static float gauss6x6_data[324]={
    0.000f,0.000f,0.062f,0.062f,0.000f,0.000f,0.000f,0.000f,0.151f,0.151f,0.000f,0.000f,0.000f,0.000f,0.236f,0.236f,0.000f,0.000f,0.000f,0.000f,0.236f,0.236f,0.000f,0.000f,0.000f,0.000f,0.151f,0.151f,0.000f,0.000f,0.000f,0.000f,0.062f,0.062f,0.000f,0.000f,
    0.000f,0.000f,0.000f,0.107f,0.029f,0.000f,0.000f,0.000f,0.003f,0.568f,0.005f,0.000f,0.000f,0.000f,0.100f,0.556f,0.000f,0.000f,0.000f,0.000f,0.556f,0.100f,0.000f,0.000f,0.000f,0.005f,0.568f,0.003f,0.000f,0.000f,0.000f,0.029f,0.107f,0.000f,0.000f,0.000f,
    0.000f,0.000f,0.000f,0.000f,0.049f,0.037f,0.000f,0.000f,0.000f,0.095f,0.307f,0.001f,0.000f,0.000f,0.063f,0.877f,0.015f,0.000f,0.000f,0.015f,0.877f,0.063f,0.000f,0.000f,0.001f,0.307f,0.095f,0.000f,0.000f,0.000f,0.037f,0.049f,0.000f,0.000f,0.000f,0.000f,
    0.000f,0.000f,0.000f,0.000f,0.000f,0.001f,0.000f,0.000f,0.000f,0.002f,0.074f,0.149f,0.000f,0.000f,0.074f,0.748f,0.336f,0.007f,0.007f,0.336f,0.748f,0.074f,0.000f,0.000f,0.149f,0.074f,0.002f,0.000f,0.000f,0.000f,0.001f,0.000f,0.000f,0.000f,0.000f,0.000f,
    0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.002f,0.028f,0.149f,0.372f,0.431f,0.232f,0.232f,0.431f,0.372f,0.149f,0.028f,0.002f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,
    0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.232f,0.431f,0.372f,0.149f,0.028f,0.002f,0.002f,0.028f,0.149f,0.372f,0.431f,0.232f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,0.000f,
    0.001f,0.000f,0.000f,0.000f,0.000f,0.000f,0.149f,0.074f,0.002f,0.000f,0.000f,0.000f,0.007f,0.336f,0.748f,0.074f,0.000f,0.000f,0.000f,0.000f,0.074f,0.748f,0.336f,0.007f,0.000f,0.000f,0.000f,0.002f,0.074f,0.149f,0.000f,0.000f,0.000f,0.000f,0.000f,0.001f,
    0.037f,0.049f,0.000f,0.000f,0.000f,0.000f,0.001f,0.307f,0.095f,0.000f,0.000f,0.000f,0.000f,0.015f,0.877f,0.063f,0.000f,0.000f,0.000f,0.000f,0.063f,0.877f,0.015f,0.000f,0.000f,0.000f,0.000f,0.095f,0.307f,0.001f,0.000f,0.000f,0.000f,0.000f,0.049f,0.037f,
    0.000f,0.029f,0.107f,0.000f,0.000f,0.000f,0.000f,0.005f,0.568f,0.003f,0.000f,0.000f,0.000f,0.000f,0.556f,0.100f,0.000f,0.000f,0.000f,0.000f,0.100f,0.556f,0.000f,0.000f,0.000f,0.000f,0.003f,0.568f,0.005f,0.000f,0.000f,0.000f,0.000f,0.107f,0.029f,0.000f
  };
  static CvMat gauss6x6[9];
  static int initialized=0;

  int i;
  if (!initialized){
    for (i=0;i<9;i++){gauss6x6[i]=cvMat(6,6,CV_32F,gauss6x6_data+i*36);}
    initialized=1;
  }

  cvScale(&gauss6x6[theta],gauss,sigma);
}

