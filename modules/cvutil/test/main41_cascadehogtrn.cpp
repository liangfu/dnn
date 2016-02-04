/**
 * @file   main35_face.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Jul  8 16:14:23 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvstageddetecthog.h"
#include "cvtimer.h"

int main(int argc, char * argv[])
{
  int i,j;
  const char * imglist[] = {
    "../data/palm.train/positive.txt",
    "../data/palm.train/negative.txt",""
  };
  char filename[2][1<<13][256];
  int count[2]={0,0};
  char line[1024];
  FILE * fp;
  IplImage * tmp;
  CvMat *** samples = new CvMat **[2];
  CvMat ** samples_stub = new CvMat *[2];

  // load image info
  for (i=0;i<2;i++){
  fp = fopen(imglist[i],"r");
  for (j=0;;j++){
    fgets(line,1024,fp);
    if (line[0]=='-'){break;}
    // strcpy(filename[i][j],line);
    sscanf(line,"%s",filename[i][j]);
  }
  fclose(fp);
  count[i]=j-1;
  }

  // load raw image
  for (i=0;i<2;i++){
  samples[i] = new CvMat * [count[i]];
  samples_stub[i] = new CvMat[count[i]];
  for (j=0;j<count[i];j++){
    if (!icvFileExist(filename[i][j])){
      LOGE("file %s not exist",filename[i][j]);
    }
    tmp = cvLoadImage(filename[i][j],0);
    samples[i][j] = cvCloneMat(cvGetMat(tmp,&samples_stub[i][j]));
    cvReleaseImage(&tmp);
  }
  }
  
  // fprintf(stderr, "INFO: %d positive samples!\n", i);

  CvStagedDetectorHOG detector;
  detector.cascadetrain(samples[0],count[0],samples[1],count[1]);

  // release raw images
  for (i=0;i<2;i++){
  for (j=0;j<count[i];j++){
    cvReleaseMat(&samples[i][j]);
  }
  delete [] samples[i];
  }
  delete [] samples;
  
  return 0;
}
