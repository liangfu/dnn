/**
 * @file   cvio_test.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Jan 10 19:32:00 2014
 * 
 * @brief  
 * 
 * 
 */
#include "compvis.h"

int cvGetFileSuffixTest();

int main()
{
  int i;
  typedef struct {int (*func)();const char * name;} CvTestFuncType;
  CvTestFuncType funcarr[]={
	{cvGetFileSuffixTest,"cvGetFileSuffix"},
	{cvGetFileSuffixTest,"cvGetFileSuffix"},
	{cvGetFileSuffixTest,"cvGetFileSuffix"},
	// ... 
	{0,""},0
  };
  for (i=0;funcarr[i].func;i++){
	if (1==funcarr[i].func[0]()){
	  fprintf(stderr,"testing %s ... [success]\n",funcarr[i].name);
	}else{
	  fprintf(stderr,"testing %s ... [fail]\n",funcarr[i].name);
	}
  }

  return 0;
}

int cvGetFileSuffixTest()
{
  int retval=0;
  const char * fullname="c:/test.tar.gz";
  char suffix[10];
  retval=cvGetFileSuffix(fullname,suffix);
  assert(retval==11);
  fprintf(stderr,"info: suffix returned as `%s`\n",suffix);
  return ((retval==11)&&(!strcmp(suffix,".gz")));
}
