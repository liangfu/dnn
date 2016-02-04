/**
 * @file   cvclassifier4fdesc.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Thu May 16 14:40:41 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_CLASSIFIER_FOR_FOURIERDESC_H__
#define __CV_CLASSIFIER_FOR_FOURIERDESC_H__

#include "cvlda.h"

class CV_EXPORTS CvClassifier4FourierDesc : public CvLDA
{
public:
  CvClassifier4FourierDesc():
      CvLDA()
  {
    static float W_data[] = {
      -8.48,5.33,13.55,-4.08,-4.05,-0.15,-0.11,6.01,-5.61,1.41,-2.77,23.32,
      -7.93,-8.20,-6.30,-2.47,-2.53,1.47,-0.36,3.55,-3.76,4.22,-1.61,
      -3.41,14.35,6.56,-2.69,-3.50,-3.04,-9.27,-9.09,9.10,-0.64,-7.06,
      -0.27,-0.33,-0.03,-12.60,-24.77,-2.59,-6.58,6.95,12.04,2.41,14.60,
      -12.38,6.99,15.42,-4.79,4.67,-1.41,13.71,6.20,1.16,-1.34,-2.32,
      24.80,-8.73,-14.08,-8.46,0.97,-1.80,1.07,4.45,0.95,-3.04,6.55,
      -4.29,-2.99,12.66,7.84,-1.76,-4.28,0.18,-12.01,-8.58,13.09,3.54,
      -12.53,-11.24,-2.23,0.04,-20.91,-13.99,-6.59,-7.58,8.30,3.91,6.28,14.42
    };
    assert(W==0);
    W = cvCreateMat(2,45,CV_32F);
    memcpy(W->data.fl,W_data,sizeof(float)*2*45);
  }
};

#endif // __CV_CLASSIFIER_FOR_FOURIERDESC_H__
