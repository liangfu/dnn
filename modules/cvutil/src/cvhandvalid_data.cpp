/**
 * @file   cvhandvalidator.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Jun 13 16:06:27 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvhandvalidator.h"

int CvHandValidator::initialize()
{
  static float W_data[4]={0,};
  static float PRIOR_data[4]={0,};

  assert(W==NULL);
  assert(PRIOR==NULL);
  W=cvCreateMat(2,2,CV_32F);
  PRIOR=cvCreateMat(2,2,CV_32F);
  memcpy(W->data.fl, W_data, sizeof(float)*4);
  memcpy(PRIOR->data.fl, PRIOR_data, sizeof(float)*4);
  return 1;
}
