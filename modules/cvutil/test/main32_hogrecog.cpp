
//////////////////////////////////////////////////////////////////////////

#include "cvext_c.h"
#include "hogrecog.h"
#include "cvtimer.h"

int main()
{
  char imagelistfn[2][256];
  sprintf(imagelistfn[0],"../dataset/palm/openTEST.txt");
  sprintf(imagelistfn[1],"../dataset/palm/closeTEST.txt");
  char line[256],fn[256];float x,y,w,h;CvRect roi;int iter;
  IplImage * img;

  for (iter=0;iter<2;iter++){
    FILE * fp = fopen(imagelistfn[iter], "r");
    fprintf(stderr, "iter:%d\n",iter);
    while(1)
    {
      fgets(line,256,fp);
      if (line[0]=='-')  {break;}
      if (line[0]=='#')  {continue;}
      sscanf(line,"%s %f %f %f %f\n",fn,&x,&y,&w,&h);
      fprintf(stderr,"%s\n",fn);
      img = cvLoadImage(fn,0);
      roi.x=x;roi.y=y;roi.width=w;roi.height=h;
CV_TIMER_START();
      int retval = hogRecog_impl(img,roi);
CV_TIMER_SHOW();
      fprintf(stderr, "result: %d\n", retval);
      assert((iter)==retval);
    }
  }
  
  return 0;
}
