/**
 * @file   cvclassifier.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Oct  9 17:54:14 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_CLASSIFIER_H__
#define __CV_CLASSIFIER_H__

#include "cvext_c.h"

#ifdef WITH_LIBSVM
#include "svm.h"

class CvClassifierLIBSVM
{
 protected:
  int m_initialized;
  struct svm_model * m_svmmodel;
  virtual int initialize()=0;

 public:

  CvClassifierLIBSVM():m_initialized(0),m_svmmodel(0)
  {
  }

  ~CvClassifierLIBSVM()
  {
    if (m_svmmodel){svm_free_and_destroy_model(&m_svmmodel);m_svmmodel=0;}
  }

  inline int initialized() {return m_initialized;}

  int predict(CvMat * sample){
    if (!initialized()){LOGE("classifier not initialized!");return 0;}
    int i,retval=-1;
    svm_node * x = (svm_node*)malloc(sizeof(svm_node)*(sample->cols+1));
    for (i=0;i<sample->cols;i++){
      x[i].index=i+1;
assert((sample->rows==1)&&(sample->cols==378));
assert(CV_MAT_TYPE(sample->type)==CV_32F);
      x[i].value=sample->data.fl[i]; 
    }x[i].index=-1;
    retval=svm_predict(m_svmmodel,x)>0;
    free(x);
    return retval;
  }
};

#endif //WITH_LIBSVM

#endif // __CV_CLASSIFIER_H__
