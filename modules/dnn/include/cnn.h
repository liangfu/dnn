/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __CNN_H__
#define __CNN_H__

// #include "cv.h"
// #include "highgui.h"

#include "cxcore.h"
#include "list.h"
#include "layers.h"

#include <limits.h>

#ifdef __cplusplus

/****************************************************************************************\
*                               Main struct definitions                                  *
\****************************************************************************************/

/* log(2*PI) */
#define CV_LOG2PI (1.8378770664093454835606594728112)

/* columns of <trainData> matrix are training samples */
#define CV_COL_SAMPLE 0

/* rows of <trainData> matrix are training samples */
#define CV_ROW_SAMPLE 1

#define CV_IS_ROW_SAMPLE(flags) ((flags) & CV_ROW_SAMPLE)

/* Variable type */
#define CV_VAR_NUMERICAL    0
#define CV_VAR_ORDERED      0
#define CV_VAR_CATEGORICAL  1

#define CV_TYPE_NAME_ML_CNN         "opencv-ml-cnn"

#if 1
#define CV_STAT_MODEL_MAGIC_VAL 0x77770000 //added by lxts
#define CV_CNN_MAGIC_VAL 0x00008888 //added by lxts
/****************************************************************************************\
*                            Convolutional Neural Network                                *
\****************************************************************************************/
typedef struct CvCNNetwork CvCNNetwork;

#define CV_CNN_NONE          0

#define CV_CNN_DELTA_W_INCREASE_FIRSTORDER  0
#define CV_CNN_DELTA_W_INCREASE_LM        1

#define CV_CNN_GRAD_ESTIM_RANDOM        0
#define CV_CNN_GRAD_ESTIM_BY_WORST_IMG  1

typedef void (CV_CDECL *CvCNNetworkAddLayer)(CvCNNetwork* network, CvCNNLayer* layer);
typedef CvCNNLayer* (CV_CDECL *CvCNNetworkGetLayer)(CvCNNetwork* network, const char * name);
typedef void (CV_CDECL *CvCNNetworkRelease)(CvCNNetwork** network);

// #define CV_STAT_MODEL_PARAM_FIELDS() int flags

typedef struct CvStatModelParams
{
  // CV_STAT_MODEL_PARAM_FIELDS();
  int flags;
} CvStatModelParams;

typedef CvCNNetwork * (CV_CDECL *CvCNNetworkRead)( CvFileStorage * fs);
typedef void (CV_CDECL *CvCNNetworkWrite)( CvCNNetwork *, CvFileStorage * fs);

typedef struct CvCNNetwork
{
  int n_layers;
  CvCNNLayer * first_layer;
  CvCNNetworkAddLayer add_layer;
  CvCNNetworkGetLayer get_layer;
  CvCNNetworkRead read;                           
  CvCNNetworkWrite write;                           
  CvCNNetworkRelease release;
}CvCNNetwork;

//add by lxts on jun-22-2008
// #define CV_STAT_MODEL_PARAM_FIELDS() CvMat * cls_labels

typedef struct CvCNNStatModelParams
{
  // CV_STAT_MODEL_PARAM_FIELDS();
  CvMat * cls_labels;
  // network must be created by the functions cvCreateCNNetwork
  // and <add_layer>
  CvCNNetwork * network;
  CvMat * etalons;
  // termination criteria
  int max_iter;
  int start_iter;
  int grad_estim_type;
  int batch_size;
}CvCNNStatModelParams;

// this macro is added by lxts on jun/22/2008
struct CvCNNStatModel;

typedef float (CV_CDECL *CvCNNStatModelPredict) (const CvCNNStatModel *,const CvMat *,CvMat *);
typedef void (CV_CDECL *CvCNNStatModelUpdate)(
        CvCNNStatModel* _cnn_model, const CvMat* _train_data, int tflag,
        const CvMat* _responses, const CvStatModelParams* _params,
        const CvMat*, const CvMat* _sample_idx,
        const CvMat*, const CvMat* );
typedef void (CV_CDECL *CvCNNStatModelRelease) (CvCNNStatModel **);

typedef struct CvCNNStatModel
{
  int flags;                                             
  CvCNNStatModelPredict predict;                         
  CvCNNStatModelUpdate update;                           
  CvCNNStatModelRelease release;
  CvCNNetwork* network;
  // etalons are allocated as rows, the i-th etalon has label cls_labeles[i]
  CvMat* etalons;
  // classes labels
  CvMat* cls_labels;
}CvCNNStatModel;

CVAPI(CvCNNetwork*) cvCreateCNNetwork( CvCNNLayer* first_layer );

CVAPI(CvCNNStatModel*) cvTrainCNNClassifier(
            const CvMat* train_data, int tflag,
            const CvMat* responses,
            const CvCNNStatModelParams* params, 
            const CvMat* CV_DEFAULT(0),
            const CvMat* sample_idx CV_DEFAULT(0),
            const CvMat* CV_DEFAULT(0), const CvMat* CV_DEFAULT(0) );

CVAPI(CvCNNetwork*) cvLoadCNNetworkModel(const char * filename);

CVAPI(CvCNNStatModelParams*) cvLoadCNNetworkSolver(const char * filename);

CVAPI(CvCNNLayer*) cvGetCNNLastLayer(CvCNNetwork * network);

/****************************************************************************************\
*                               Estimate classifiers algorithms                          *
\****************************************************************************************/

CVAPI(CvCNNStatModel*) cvCreateCNNStatModel(int flag, int size);

#endif /* 1 */

#endif /* __cplusplus */

#endif /*__CNN_H__*/
/* End of file. */
