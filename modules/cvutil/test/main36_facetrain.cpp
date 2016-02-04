/**
 * @file   main35_face.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Jul  8 16:14:23 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvstageddetecthaar.h"
// #include "cvfacedetector.h"
#include "cvtimer.h"

int main(int argc, char * argv[])
{
  int i,j;
  const char * imglist[] = {
    "../data/face.train/positive.txt",
    "../data/face.train/negative.txt",""
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

  CvStagedDetectorHaar detector;

#if 1
  detector.cascadetrain(samples[0],count[0],samples[1],count[1],0.7,0.94,0.001);

#else

  const char * fnlist[] = {
    "../data/gesturePalm_JC2-0000.png",
    "../data/gesturePalm_JC3train-204.png",
    "../data/gesturePalm_Jiang-0000.png",
    "../data/gesturePalm_Liulu-0300.png",
    "../data/gesturePalm_Steven-0000.png",
    "../data/mei-family.png",
    "",
  };

  for (j=0;strlen(fnlist[j])>0;j++)
  {
    IplImage * raw = cvLoadImage(fnlist[j],0);
    CvMat img_stub;
    CvMat * img = cvGetMat(raw,&img_stub);
    CvRect rois[1000];
CV_TIMER_START();
    int nfaces = detector.detect(img,rois);
CV_TIMER_SHOW();
    for (i=0;i<nfaces;i++){
      cvRectangle(img,cvPoint(rois[i].x,rois[i].y),
                  cvPoint(rois[i].x+rois[i].width,rois[i].y+rois[i].height),
                  cvScalarAll(255));
    }
    char label[100];sprintf(label,"nfaces: %d",nfaces);
    cvDrawLabel(img,label,cvScalarAll(128));
    // if (img->cols>1250)
    // {
    //   double scale = 1250./float(img->cols);
    //   CvMat * resized=cvCreateMat(img->rows*scale,img->cols*scale,CV_8U);
    //   cvResize(img,resized);
    //   cvShowImage("Test",resized); CV_WAIT();
    //   cvReleaseMat(&resized);
    // }else
    {
      cvShowImage("Test",img); CV_WAIT();
    }
    cvReleaseImage(&raw);
  }
#endif

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
