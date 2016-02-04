/**
 * @file   cvhandvalidator.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu Jun 13 16:06:34 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_HAND_VALIDATOR_H__
#define __CV_HAND_VALIDATOR_H__

#include "cvext_c.h"
#include "cvlda.h"

class CV_EXPORTS CvHandValidator : public CvLDA
{
  int initialize();
public:
  CvHandValidator(){
    if (!initialize()){
      fprintf(stderr, "WARNING: fail to initialize hand validator.\n");
    }
  }
  ~CvHandValidator(){}

  int validate(CvMat * imgY);
};

#endif // __CV_HAND_VALIDATOR_H__
