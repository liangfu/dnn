//
//  gpumat.h
//  DNN
//
//  Created by Liangfu Chen on 8/5/16.
//
//

#ifndef __GPUMAT_H__
#define __GPUMAT_H__

#include "cxcore.h"
#ifdef __APPLE__
#include <OpenCL/CL.h>
#else
#include <CL/CL.h>
#endif

typedef unsigned char uchar;

typedef struct {
  cl_device_id device;
  cl_uint row_alignment;
} CvGpuDevice;

CvGpuDevice * cvCreateGpuDevice(cl_device_id gpu_id){
  CvGpuDevice * gpu = new CvGpuDevice;
  gpu->device = gpu_id;
  gpu->row_alignment = requiredOpenCLAlignment(gpu->device);
  
  return gpu;
}

void cvReleaseGpuDevice(CvGpuDevice ** gpu){delete (*gpu);}

typedef struct {
  int type;
  int rows;
  int cols;
  union{
    uchar * ptr;
    float * fl;
  } data;
  cl_mem device;
} CvGpuMat;

CvGpuMat cvGpuMat(int rows, int cols, int type, void * ptr){
  CvGpuMat mat;
  mat.type=type;mat.rows=rows;mat.cols=cols;mat.data.ptr=(uchar*)ptr;
  return mat;
}

void cvGpuMatUpload(CvMat * src, CvGpuMat * dst){}
void cvGpuMatDownload(CvGpuMat * src, CvMat * dst){}

void cvGpuGEMM(CvGpuMat * A, CvGpuMat * B, float alpha, CvGpuMat * C, float beta, int opt)
{
  //CvGpuMat src;
  
}



#endif /* __GPUMAT_H__ */
