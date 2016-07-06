#include "dnn.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  i | input   | data.xml.gz   | location of input data file  }"
          "{  l | label   | data.txt      | location of label file  }"
          "{  s | start   | 0             | start index  }"
          "{  h | help    | false         | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
  const char * input_filename = parser.get<string>("input").c_str();
  const char * label_filename = parser.get<string>("label").c_str();
  const int start_index = parser.get<int>("start");
  
  CvMat * input = (CvMat*)cvLoad(input_filename);
  CvMat * label = cvCreateMat(input->rows,1,CV_32S);
  FILE * fp = fopen(label_filename,"rt"); char tstr[8];
  for (int ii=0;ii<input->rows;ii++){
    fgets(tstr,sizeof(tstr),fp);
    CV_MAT_ELEM(*label,int,ii,0)=atoi(tstr);
  }
  fclose(fp);

  const int psize = 10;
  CvMat * disp = cvCreateMat(64*psize,64,CV_32F);
  for (int ii=start_index;ii<input->rows;ii+=psize){
    CvMat input_submat_hdr,input_submat_hdr2;
    cvGetRows(input,&input_submat_hdr,ii,ii+psize);
    cvReshape(&input_submat_hdr,&input_submat_hdr2,0,disp->rows);
    cvCopy(&input_submat_hdr2,disp);
    for (int jj=0;jj<psize;jj++){
      sprintf(tstr,"%d",ii+jj+1);
      cvPutTextEx(disp,tstr,cvPoint(0,jj*64+10),CV_WHITE,.4);
      sprintf(tstr,"%d",CV_MAT_ELEM(*label,int,ii+jj,0));
      cvPutTextEx(disp,tstr,cvPoint(44,jj*64+10),CV_WHITE,.4);
    }
    CV_SHOW(disp);
  }
  cvReleaseMat(&disp);
  
  cvReleaseMat(&input);
  cvReleaseMat(&label);
  
  return 0;
}
