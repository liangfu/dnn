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

int CvHandValidator::validate(CvMat * imgYpatch)
{
  // if (!hog){
  //   hog = cvCreateMat(1,9,CV_32F);
  // }

  float result_data[2]={-1,-1};
  CvMat result = cvMat(1,2,CV_32F,result_data);
  // predict_withprior(hog,&result);
  return 1;
}
