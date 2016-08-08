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
  cl_context context;
  cl_command_queue command_queue;
  cl_uint row_alignment;
  cl_ulong max_device_work_group_size;
  // cl_ulong max_alloc_size;
  // cl_ulong max_global_mem_size;
} CvGpuDevice;

typedef struct {
  cl_uint max_kernel_work_group_size;
} CvGpuKernel;

int cvGetGpuDeviceList(cl_device_id ** devices)
{
  /* Step1: Getting platforms and choose an available one.*/
  cl_uint numPlatforms;	//the NO. of platforms
  cl_platform_id platform = NULL;	//the chosen platform
  cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS){
    fprintf(stderr,"Error: Getting platforms!\n");return 0;
  }
  
  /* For clarity, choose the first available platform. */
  if(numPlatforms > 0){
    cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    platform = platforms[0]; free(platforms);
  }
  
  /* Step 2:Query the platform and choose the first GPU device if has one.
     Otherwise use the CPU as device.*/
  cl_uint				numDevices = 0;
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (numDevices == 0){	//no GPU available.
    fprintf(stderr,"No GPU device available.\nChoose CPU as default device.\n");
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
    *devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, *devices, NULL);
  }else{
    *devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, *devices, NULL);
  }
  return numDevices;
}

CvGpuDevice * cvCreateGpuDevice(cl_device_id device_id){
  CvGpuDevice * gpu = new CvGpuDevice;
  gpu->device = device_id;
  gpu->context = clCreateContext(NULL, 1, &gpu->device, NULL, NULL, NULL);
  gpu->command_queue = clCreateCommandQueue(gpu->context, gpu->device, 0, NULL);
  gpu->row_alignment = requiredOpenCLAlignment(gpu->device);
  gpu->max_device_work_group_size = deviceMaxWorkGroupSize(gpu->device);  
  return gpu;
}

void cvReleaseGpuDevice(CvGpuDevice ** gpu){delete (*gpu);}

typedef struct {
  int type;
  int rows;
  int cols;
  union{
    uchar * ptr;
    short * s;
    int   * i;
    float * fl;
    double * db;
  } host;
  cl_mem device_mem;
  size_t stride;
  size_t matrix_memory_size;
} CvGpuMat;

CvGpuMat cvGpuMat(CvGpuDevice * gpu, int rows, int cols, int type, void * ptr, cl_mem_flags flags){
  CvGpuMat mat; cl_int err=0;
  mat.type=type;mat.rows=rows;mat.cols=cols;
  
  mat.host.ptr=(uchar*)ptr; // apply aligned host memory
  
  mat.stride = (cols*sizeof(float) + gpu->row_alignment - 1) & ~(gpu->row_alignment - 1);
  //fprintf(stdout,"Memory row stride to ensure necessary alignment: %lu bytes\n",mat.stride);
  mat.stride /= sizeof(float); assert(cols <= mat.stride); // calculate row stride in elements of T
  mat.matrix_memory_size = rows*mat.stride*sizeof(float);
  //fprintf(stdout,"Size of memory region for one matrix: %lu bytes\n",mat.matrix_memory_size);
  mat.device_mem=clCreateBuffer(gpu->context, flags, mat.matrix_memory_size, ptr, &err);
  SAMPLE_CHECK_ERRORS(err);
  return mat;
}

void cvGpuMatUpload(CvMat * src, CvGpuMat * dst){}
void cvGpuMatDownload(CvGpuMat * src, CvMat * dst){}

typedef struct {
  CvGpuKernel * kernel;
  CvGpuMat * matrix_A;
  CvGpuMat * matrix_B;
  CvGpuMat * matrix_C;
} CvGpuGEMMKernel;

void cvGpuGEMM(CvGpuMat * A, CvGpuMat * B, float alpha, 
               CvGpuMat * C, float beta, int opt)
{
  CV_FUNCNAME("cvGpuGEMM");
  __CV_BEGIN__;
  //CV_ASSERT(A->rows==C->rows && A->cols==B->rows && B->cols==C->cols);
  //int M = A->rows, K = A->cols, N = B->cols;
  
  __CV_END__;
}

void gemm_v2(const char * program_text, const string build_options)
{
  const int tile_size_M = 1;
  const int tile_group_M = 8;
  const int tile_size_N = 16;
  const int tile_group_N = 1;
  const int tile_size_K = 16;
  const int M = 1024, K = 1024, N = 1024;
  CvRNG rng = cvRNG(-1);
  
  CvMat * A_host = cvCreateMat(M,K,CV_32F); 
  CvMat * B_host = cvCreateMat(K,N,CV_32F);
  CvMat * C_host = cvCreateMat(M,N,CV_32F);
  cvRandArr(&rng,A_host,CV_RAND_UNI,cvScalar(-1),cvScalar(1));
  cvRandArr(&rng,B_host,CV_RAND_UNI,cvScalar(-1),cvScalar(1));
  cvZero(C_host);

  cl_device_id * devices = NULL;
  int n_devices = cvGetGpuDeviceList(&devices);
  if (n_devices<1){return;}
  CvGpuDevice * device = cvCreateGpuDevice(devices[0]);
  
  // Ensures that each matrix memory row is aligned
  fprintf(stdout,"Running `gemm_nn` kernel with matrix size: %dx%d\n",M,N);
  
  CvGpuMat A_device = cvGpuMat(device,A_host->rows,A_host->cols,A_host->type,A_host->data.ptr,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR);
  CvGpuMat B_device = cvGpuMat(device,B_host->rows,B_host->cols,B_host->type,B_host->data.ptr,CL_MEM_READ_ONLY|CL_MEM_HOST_PTR);
  CvGpuMat C_device = cvGpuMat(device,C_host->rows,C_host->cols,C_host->type,C_host->data.ptr,CL_MEM_READ_WRITE|CL_MEM_HOST_PTR);
  
  cvReleaseMat(&A_host);
  cvReleaseMat(&B_host);
  cvReleaseMat(&C_host);
}

#endif /* __GPUMAT_H__ */
