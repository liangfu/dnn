/**
 * @file   cvstageddetectorhog.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Sep 10 15:46:21 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvstageddetecthog.h"

CvMat * icvGenerateFeaturesHOG(int nr, int nc);

int CvStagedDetectorHOG::detect(CvMat * img, CvRect ROIs[])
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

int CvStagedDetectorHOG::
train_ada(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg, int iter)
{
  int i,j,ii,jj,k,nr=posimgs[0]->rows,nc=posimgs[0]->cols;
  static const int maxiter=8000;
  int m=nneg,l=npos;
  int nsamples = m+l;
  int count[2]={m,l};
  if (!m_features) { m_features = icvGenerateFeaturesHOG(nr,nc); }
  int nfeatures = m_features->rows;
  if (!m_weights) {
    m_weights = cvCreateMat(maxiter,nsamples,CV_64F);
    for (i=0;i<2;i++){
    for (j=0;j<count[i];j++){
      m_weights->data.db[i*count[0]+j]=1./double(count[i]);
    }
    }
  }

  CvMat ** magni; CvMat ** angle;
  CvMat ** evalres_precompute;
  {
    
  }
  if (!m_feature_precomputed)
  {
    feature_precompute_hog(posimgs,count[1],negimgs,count[0],magni,angle);
    // evaluate(magni,angle,evalres_precompute);
    m_feature_precomputed=1;
  }
  
  // for (iter=0;iter<maxiter;iter++)
  {
    CvMat * epsilon = cvCreateMat(1,nfeatures,CV_64F);

    // normalize weights
    {
      double * wtptr = m_weights->data.db+iter*m_weights->cols;
      double wtsum=0;
      for (i=0;i<m_weights->cols;i++){ wtsum+=wtptr[i]; }
      double invwtsum=1./wtsum;
      for (i=0;i<m_weights->cols;i++){ wtptr[i]*=invwtsum; }
    }
    
    cvReleaseMat(&epsilon);
  }

  // release memory ..
  if (m_feature_precomputed){
    for (i=0;i<nsamples;i++){
      cvReleaseMat(&magni[i]);
      cvReleaseMat(&angle[i]);
    }
  }
  return 1;
}

int CvStagedDetectorHOG::
validate(int ni, double & fi, double & di)
{
  return 1;
}

int CvStagedDetectorHOG::
adjust(int ni, double dtar, double & fi, double & di)
{
  return 1;
}

// cascade detector training framework 
int CvStagedDetectorHOG::
cascadetrain(CvMat ** posimgs, int npos, CvMat ** negimgs, int nneg,
             double fper, double dper, double ftarget)
{
  static const int train_type=1;
  static trainfunctype trainfuncarr[2]={
    &CvStagedDetectorHOG::train_svm,
    &CvStagedDetectorHOG::train_ada
  };
  int i,ni,maxiter=200;
  CvMat * frate = cvCreateMat(1,maxiter,CV_64F); frate->data.db[0]=1.0;
  CvMat * drate = cvCreateMat(1,maxiter,CV_64F); drate->data.db[0]=1.0;

  for (i=0;(frate->data.db[i]>ftarget)&&(i<maxiter);)
  {
    i++;
    double fi=frate->data.db[i-1],di;
    for (ni=0;fi>frate->data.db[i-1]*fper;){
      ni++;

      // adaboost training
      ((*this).*trainfuncarr[train_type])(posimgs,npos,negimgs,nneg,ni);

      // validation
      validate(ni,fi,di);

      // adjust threshold for i-th classifier
      adjust(ni,dper*drate->data.db[i-1],fi,di);

      // update weights
      // update_weidhts();
      
      // validation
      validate(ni,fi,di);
    }
    frate->data.db[i]=fi;
    drate->data.db[i]=di;

    // add to negative training set ...
  }
  
  cvReleaseMat(&frate);
  cvReleaseMat(&drate);
  
  return 1;
}

int CvStagedDetectorHOG::
feature_precompute_hog(CvMat ** posimgs, int npos,
                       CvMat ** negimgs, int nneg,
                       CvMat ** magni, CvMat ** angle)
{
  int nsamples = npos+nneg;
  int i,j,nr=posimgs[0]->rows,nc=posimgs[0]->cols,jj,ii;
  int count[2]={nneg,npos};
  // allocate memory
  {
    magni = new CvMat *[nsamples];
    angle = new CvMat *[nsamples];
    for (i=0;i<nsamples;i++){
      magni[i]=cvCreateMat(nr,nc,CV_32F); cvZero(magni[i]);
      angle[i]=cvCreateMat(nr,nc,CV_8U);  cvZero(angle[i]);
    }
  }
  // precompute magni and angle for all samples
  {
    CvMat * dx = cvCreateMat(nr,nc,CV_16S);
    CvMat * dy = cvCreateMat(nr,nc,CV_16S);
    int idx;
    // for (i=0;i<2;i++){
    for (i=1;i>-1;i--){
    // magni=cvCreateMat(nr,nc,CV_32F);
    // angle=cvCreateMat(nr,nc,CV_8U);  
    for (j=0;j<count[i];j++){
      idx=count[0]*i+j;
      cvSobel((i==0)?negimgs[j]:posimgs[j],dx,1,0,1);
      cvSobel((i==0)?negimgs[j]:posimgs[j],dy,0,1,1);
      // magni=dx^2+dy^2; angle=atan2(dy,dx);
      int dxstep,dystep,magstep,angstep;
      dxstep=dx->cols;dystep=dy->cols;
      magstep=magni[idx]->cols;angstep=angle[idx]->cols;
      short * dxptr = dx->data.s+dxstep;
      short * dyptr = dy->data.s+dystep;
      float * magptr = magni[idx]->data.fl +magstep;
      uchar * angptr = angle[idx]->data.ptr+angstep;
      double tmpval;
      for (ii=1;ii<nr-1;ii++){
      for (jj=1;jj<nc-1;jj++){
        magptr[jj]=cvSqrt(dxptr[jj]*dxptr[jj]+dyptr[jj]*dyptr[jj]);
        tmpval=icvFastAtan2(dyptr[jj],dxptr[jj]);
        tmpval=(tmpval<0)?-tmpval:tmpval;
        angptr[jj]=cvFloor(tmpval*9./3.1416);
      }
      dxptr+=dxstep;dyptr+=dystep;magptr+=magstep;angptr+=angstep;
      }
      CV_SHOW(magni[idx]);
      CV_SHOW(angle[idx]);
      // break;
    }
    // break;
    }
    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
  }
}

// hog parameters: 
CvMat * icvGenerateFeaturesHOG(int nr, int nc)
{
  typedef float hogpar[4];
  int log2count=8,count=0;
  hogpar * buffer = (hogpar*)malloc(sizeof(hogpar)*(1<<log2count));
  int i,j,celliter,griditer;
  // for (i=0;i<nr;i+=2){
  // for (j=0;j<nc;j+=2){
  for (i=8;i<nr-8;i+=2){
  for (j=8;j<nc-8;j+=2){
  for (celliter=4;celliter<10;celliter+=2){ // 4,6,8
    if ((1<<log2count)<=count+10){
      buffer = (hogpar*)realloc(buffer,sizeof(hogpar)*(1<<(++log2count)));
    }
    if ((i+celliter<nr)&&(j+celliter<nc)){
      hogpar tmp={i,j,celliter,celliter};
      memcpy(buffer[count++],tmp,sizeof(hogpar));
    }
    if ((i+celliter*2<nr)&&(j+celliter<nc)){
      hogpar tmp={i,j,celliter,celliter*2};
      memcpy(buffer[count++],tmp,sizeof(hogpar));
    }
    if ((i+celliter<nr)&&(j+celliter*2<nc)){
      hogpar tmp={i,j,celliter*2,celliter};
      memcpy(buffer[count++],tmp,sizeof(hogpar));
    }
  }    
  }
  }
  CvMat * features =
      cvCreateMat(count,sizeof(hogpar)/sizeof(float),CV_32F);
  memcpy(features->data.ptr,buffer,sizeof(hogpar)*count);
  free(buffer);
  return features;
}

