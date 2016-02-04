/**
 * @file   main34_rawiv.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Thu Jul  4 09:56:34 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvext_io.h"

int main(int argc, char * argv[])
{
  CvMatND * data = icvLoadRAWIV(argv[1]);
  int i,dims[3]={data->dim[0].size,data->dim[1].size,data->dim[2].size};
  CvMat * slice =
      cvCreateMat(dims[1],dims[2],CV_MAT_TYPE(data->type));
  int bytes=CV_MAT_TYPE(data->type)==CV_32F?4:1;
  if (bytes==1)
  for (i=0;i<dims[0];i++){
    memcpy(slice->data.ptr,data->data.ptr+dims[1]*dims[2]*i,
           dims[1]*dims[2]);
    CV_SHOW(slice);
  }
  else if (bytes==4)
  for (i=0;i<dims[0];i++){
    memcpy(slice->data.ptr,data->data.fl+dims[1]*dims[2]*i,
           4*dims[1]*dims[2]);
    CV_SHOW(slice);
  }
  cvReleaseMatND(&data);
  cvReleaseMat(&slice);
  return 0;
}

