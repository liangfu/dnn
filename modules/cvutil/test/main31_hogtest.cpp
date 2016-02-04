/**
 * @file   main30_hogtrain.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jun 21 17:39:45 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvhog.h"
#include "cvlda4hog.h"

#define MAX_SAMPLES 2000
#define WITH_TZG 1

void cvPrepareGradientROI(CvMat * mat, CvMat * dx, CvMat * dy,
                          float scale, CvRect roi);

int main(int argc, char * argv[])
{
  typedef struct {char buffer[512];char fn[256];float x,y,w,h;} CvTrainData;
  static float hogfeature_data[378];
  static float result_data[2];
  static CvLDA4HOG lda4hog;

  char imagelistfn[2][256];
  sprintf(imagelistfn[0],"../dataset/palm/open.txt");
  sprintf(imagelistfn[1],"../dataset/palm/close.txt");

  int statusiter=0;
  for (statusiter=0;statusiter<2;statusiter++)
  {

  int i,j,N;
  CvTrainData dat[MAX_SAMPLES];
  {
  FILE * fp = fopen(imagelistfn[statusiter], "r");
  for (i=0;i<MAX_SAMPLES;i++) {
    fgets(dat[i].buffer,512,fp);
    if (dat[i].buffer[0]=='-'){break;}
    if (dat[i].buffer[0]=='#'){i--;continue;}
    sscanf(dat[i].buffer, "%s %f %f %f %f\n",
           dat[i].fn,&dat[i].x,&dat[i].y,&dat[i].w,&dat[i].h);
#if WITH_TZG
    dat[i].x=dat[i].x+cvRound(dat[i].w*.15);
    dat[i].y=dat[i].y+cvRound(dat[i].h*.15);
    dat[i].w=cvRound(dat[i].w*.7);
    dat[i].h=cvRound(dat[i].h*.7);
#endif
  }
  N=i;
  fprintf(stderr, "INFO: %d training dat collected!\n", i);
  fclose(fp);
  }

  CvMat hogfeature = cvMat(1,378,CV_32F,hogfeature_data);
  CvMat result = cvMat(1,2,CV_32F,result_data);
  
  int hogsizes[]={7,6,9};
  int hogszprod = hogsizes[0]*hogsizes[1]*hogsizes[2];
  for (i=0;i<N;i++)
  {
    IplImage * img = cvLoadImage(dat[i].fn,0);
    CvMat mat_stub;
    CvMat * mat = cvGetMat(img,&mat_stub);
    CvMatND * hog = cvCreateMatND(3,hogsizes,CV_32F);
    int nr=mat->rows,nc=mat->cols;
    CvMat * dx = cvCreateMat(nr,nc,CV_16S);
    CvMat * dy = cvCreateMat(nr,nc,CV_16S);
    CvMat * magni = cvCreateMat(nr,nc,CV_32F); cvZero(magni);
    CvMat * angle = cvCreateMat(nr,nc,CV_32F); cvZero(angle);
    float warp_p_data[]={dat[i].h/42.,dat[i].x-3.,dat[i].y-3.};
    CvMat warp_p = cvMat(3,1,CV_32F,warp_p_data);

    // cvSobel(img,dx,1,0,1);
    // cvSobel(img,dy,0,1,1);
    cvPrepareGradientROI(mat, dx, dy, warp_p_data[0],
                         cvRect(dat[i].x,dat[i].y,dat[i].w,dat[i].h));
    cvSet(magni,cvScalar(-1));

    icvCalcWarpHOG(mat,&warp_p,hog,6,2,dx,dy,magni,angle);
    // for (j=0;j<hogszprod;j++){ fprintf(stderr, "%ff,", hog->data.fl[j]); }
    {
      memcpy(hogfeature_data,hog->data.fl,sizeof(float)*378);
      lda4hog.predict_withprior(&hogfeature,&result);
      fprintf(stderr, "result: %f,fn: %s\n", result_data[0],dat[i].fn);
    }
    // icvShowHOG(hog); CV_WAIT();

    // icvShowHOG(hog); CV_WAIT();
    // fprintf(stderr, "scale: %f\n", warp_p_data[0]);
    if (argc>1) { if (!strcmp(argv[1],"-show") // &&
                      // ((result_data[0]>0.4 && statusiter==1)||
                      //  (result_data[0]<0.6 && statusiter==0))
                      ){
      cvRectangle(mat,cvPoint(dat[i].x,dat[i].y),
                  cvPoint(dat[i].x+dat[i].w,dat[i].y+dat[i].h),CV_RED);
      fprintf(stderr, "INFO: display %s\n",dat[i].fn);
      CV_SHOW(mat);
      }
    }
    // CV_SHOW(magni);
    // CV_SHOW(angle);

    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
    cvReleaseMat(&magni);
    cvReleaseMat(&angle);
    cvReleaseMatND(&hog);
  }
  }  
  
  return 0;
}

void cvPrepareGradientROI(CvMat * mat, CvMat * dx, CvMat * dy,
                          float scale, CvRect roi)
{
  int nr=mat->rows,nc=mat->cols;
  cvZero(dx);cvZero(dy);
  int step = dx->step/sizeof(short);
  assert(step==dy->step/sizeof(short));
  assert(step==mat->step/sizeof(uchar));
  int ystart= MAX(1,roi.y-6*scale-1);
  int yend  = MIN(nr-1,roi.y+6*scale+roi.height-1);
  int xstart= MAX(1,roi.x-6*scale+1);
  int xend  = MIN(nc-1,roi.x+6*scale+roi.width+1);
  int ystartstep=ystart*step;
  uchar * imgptr = mat->data.ptr+ystartstep;
  short * dxptr = dx->data.s+ystartstep; 
  short * dyptr = dy->data.s+ystartstep;
  double sum=0,mu=0;int count=0;
  int i,j;
  // calculate mean value in region
  for (i=ystart;i<yend;i++){
    for (j=xstart;j<xend;j++){sum+=imgptr[j];count++;}
    imgptr+=step;
  }
  mu=sum/double(count);

  // calculate variance of region
  imgptr = mat->data.ptr+ystartstep;
  sum=0;
  for (i=ystart;i<yend;i++){
    for (j=xstart;j<xend;j++){sum+=pow(imgptr[j]-mu,2);}
    imgptr+=step;
  }
  double sqvar = sqrt(sum/double(count));
  double invar = 60./sqvar;
  // fprintf(stderr, "var: %.1f,%.1f,mu:%.1f    ",sqvar,invar,mu);

  // calculate gradient with intensity value normalization
  imgptr = mat->data.ptr+ystartstep;
  for (i=ystart;i<yend;i++){
    for (j=xstart;j<xend;j++){
      dxptr[j]=(imgptr[j+1]-imgptr[j-1])*invar;
      dyptr[j]=((imgptr+step)[j]-(imgptr-step)[j])*invar;
    }
    imgptr+=step;dxptr+=step;dyptr+=step;
  }
}
