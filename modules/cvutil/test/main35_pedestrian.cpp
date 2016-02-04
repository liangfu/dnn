/**
 * @file   main34_pedestrian.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Jul  8 13:31:07 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvext_c.h"

int main()
{
  const char * listfn = "../dataset/pedestrians128x64/list.txt";
  char line[1024],imagefn[1024];int i;
  FILE * fp = fopen(listfn,"r");
  if (!fp){
    fprintf(stderr, "ERROR: fail to load file %s\n",listfn);return 0;
  }

  for (i=0;;i++)
  {
    fgets(line,1024,fp);
    if (line[0]=='-') {break;}
    sscanf(line,"%s\n",imagefn);
    IplImage * raw = cvLoadImage(imagefn,0);
    CvMat img_stub;
    CvMat * img = cvGetMat(raw,&img_stub);
    int nr = img->rows, nc = img->cols;
    CvMat * dx = cvCreateMat(nr,nc,CV_16S);
    CvMat * dy = cvCreateMat(nr,nc,CV_16S);
    float warp_p_data[3];
    CvMat warp_p = cvMat(3,1,CV_32F,warp_p_data);
    cvSobel(img,dx,1,0,1);
    cvSobel(img,dy,0,1,1);
    // cvCalcWarpHOG(img,&warp_p,);

    CV_SHOW(img);

    cvReleaseMat(&dx);
    cvReleaseMat(&dy);
    // cvReleaseMat(&magni);
    // cvReleaseMat(&angle);
  }  

  fclose(fp);
  return 0;
}
