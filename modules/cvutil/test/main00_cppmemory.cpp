/**
 * @file   main00_cppmemory.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Wed Sep  4 15:39:19 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvext_c.h"

#define LOG()                                   \
  fprintf(stderr,"%s: %d\n",__func__,__LINE__)

class A{public:A(){LOG();}~A(){LOG();}};
// class B:public A{public:B():A(){LOG();}~B(){LOG();}};
class B:public A{public:B():A(){LOG();}};

int main()
{
  B b;
  return 0;
}
