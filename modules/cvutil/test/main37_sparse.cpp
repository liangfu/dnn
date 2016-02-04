/**
 * @file   main37_sparse.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Fri Jul 19 16:58:15 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvsparsecoding.h"

int main()
{
  CvSparseLearner learner;
  const char * filelist[]={"../data/lena.png",""};
  learner.learn(filelist,1);
  return 0;
}
