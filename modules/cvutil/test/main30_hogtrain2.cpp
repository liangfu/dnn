/**
 * @file   main30_hogtrain.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jun 21 17:39:45 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext_c.h"
#include "hogrecog.h"

#define MAX_SAMPLES 2000
// #define WITH_TZG 1

int main(int argc, char * argv[])
{
  typedef struct {char buffer[512];char fn[256];float x,y,w,h;} CvTrainData;
  char imagelistfn[2][256];
  sprintf(imagelistfn[0],"../dataset/palm/open.txt");
  sprintf(imagelistfn[1],"../dataset/palm/close.txt");
  char featurefn[2][256];
  sprintf(featurefn[0],"../data/open.bin");
  sprintf(featurefn[1],"../data/close.bin");

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
// #if WITH_TZG
//     dat[i].x=dat[i].x+cvRound(dat[i].w*.15);
//     dat[i].y=dat[i].y+cvRound(dat[i].h*.15);
//     dat[i].w=cvRound(dat[i].w*.7);
//     dat[i].h=cvRound(dat[i].h*.7);
// #endif
  }
  fprintf(stderr, "INFO: %d training dat collected!\n", i);
  fclose(fp);
  }
  N=i;

  float * hogfts_data = new float[378*MAX_SAMPLES];
  CvMat hogfts = cvMat(MAX_SAMPLES,378,CV_32F,hogfts_data);
  
  int hogsizes[]={7,6,9};
  int hogszprod = hogsizes[0]*hogsizes[1]*hogsizes[2];
  for (i=0;i<N;i++)
  {
    IplImage * img = cvLoadImage(dat[i].fn,0);
    CvMatND * hog = cvCreateMatND(3,hogsizes,CV_32F);

    hogRecog(img,cvRect(dat[i].x,dat[i].y,dat[i].w,dat[i].h),hog);
    memcpy(hogfts_data+378*i,hog->data.fl,sizeof(float)*378);
// fprintf(stderr,"%f,%f\n",(hogfts_data+378*i)[0],(hogfts_data+378*i)[1]);
    
    cvReleaseMatND(&hog);
  }

  // save feature into binary file for MATLAB training
  {
    FILE * fp=fopen(featurefn[statusiter],"wb");
    fwrite(hogfts_data,4,378*i,fp);
    fclose(fp);
  }
  delete [] hogfts_data;

  }
  
  return 0;
}
