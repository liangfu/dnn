/**
 * @file   cvsvm4hog.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Oct  9 18:08:14 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_SVM_FOR_HOG_H__
#define __CV_SVM_FOR_HOG_H__

#include "cvclassifier.h"

#ifdef WITH_LIBSVM
class CvSVM4HOG : public CvClassifierLIBSVM
{
 protected:
  int initialize();

 public:
  CvSVM4HOG():CvClassifierLIBSVM()
  {
    initialize();
  }

  ~CvSVM4HOG()
  {
  }
};
#endif //WITH_LIBSVM

#endif // __CV_SVM_FOR_HOG_H__



