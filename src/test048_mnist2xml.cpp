#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

int ReverseInt (int i)
{
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
 
CvMat * read_Mnist_Images(char * filename)
{
  FILE * fp = fopen(filename,"r");
  if (!fp){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  int magic_number = 0;
  int number_of_images = 0;
  int n_rows = 0;
  int n_cols = 0;
  fread((char*) &magic_number, sizeof(magic_number),1,fp);
  magic_number = ReverseInt(magic_number);
  fread((char*) &number_of_images,sizeof(number_of_images),1,fp);
  number_of_images = ReverseInt(number_of_images);
  fread((char*) &n_rows, sizeof(n_rows),1,fp);
  n_rows = ReverseInt(n_rows);
  fread((char*) &n_cols, sizeof(n_cols),1,fp);
  n_cols = ReverseInt(n_cols);
  CvMat * data = cvCreateMat(number_of_images,n_rows*n_cols,CV_32F);
  for(int i = 0; i < number_of_images; ++i){
  for(int r = 0; r < n_rows; ++r){
  for(int c = 0; c < n_cols; ++c){
	unsigned char temp = 0;
	fread((char*) &temp, sizeof(temp),1,fp);
	CV_MAT_ELEM(*data,float,i,r*n_cols+c)=float(temp);
  }
  }
  }
  return data;
}
 
CvMat * read_Mnist_Labels(char * filename)
{
  FILE * fp = fopen(filename,"r");
  if (!fp){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  int magic_number = 0;
  int number_of_labels = 0;
  fread((char*) &magic_number, sizeof(magic_number),1,fp);
  magic_number = ReverseInt(magic_number);
  fread((char*) &number_of_labels,sizeof(number_of_labels),1,fp);
  number_of_labels = ReverseInt(number_of_labels);
  CvMat * data = cvCreateMat(number_of_labels,1,CV_8U);
  for(int i = 0; i < number_of_labels; ++i){
	unsigned char temp = 0;
	fread((char*) &temp, sizeof(temp),1,fp);
	CV_MAT_ELEM(*data,uchar,i,0)=float(temp);
  }
  return data;
}

void cvPrepareResponse(CvMat * response, CvMat * responseMat);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  s | solver  |       | location of solver file      }"
          "{  h | help    | false | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
  const char * solver_filename  = parser.get<string>("solver").c_str();
  CvNetwork * cnn = new CvNetwork();
  cnn->loadSolver(solver_filename);
  const char * training_filename_xml = cnn->solver()->training_filename();
  const char * response_filename_xml = cnn->solver()->response_filename();
  const char *  testing_filename_xml  = cnn->solver()->testing_filename();
  const char * expected_filename_xml = cnn->solver()->expected_filename();
  char training_filename[1<<10]={0,}; 
  char response_filename[1<<10]={0,}; 
  char  testing_filename[1<<10]={0,}; 
  char expected_filename[1<<10]={0,}; 

  cvGetBaseName((char*)training_filename_xml,(char*)training_filename);
  cvGetBaseName((char*)response_filename_xml,(char*)response_filename);
  cvGetBaseName((char*) testing_filename_xml,(char*) testing_filename);
  cvGetBaseName((char*)expected_filename_xml,(char*)expected_filename);

  const int nclasses = 10;
  CvMat * training    = read_Mnist_Images((char*)training_filename);
  CvMat * response    = read_Mnist_Labels((char*)response_filename);
  CvMat * testing     = read_Mnist_Images((char*)testing_filename);
  CvMat * expected    = read_Mnist_Labels((char*)expected_filename);
  CvMat * responseMat = cvCreateMat(response->rows,nclasses,CV_32F);
  CvMat * expectedMat = cvCreateMat(expected->rows,nclasses,CV_32F);
  assert(training->rows==response->rows && testing->rows==expected->rows);

  cvPrepareResponse(response,responseMat);
  cvPrepareResponse(expected,expectedMat);

  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,responseMat);
  cvSave( testing_filename_xml,testing );
  cvSave(expected_filename_xml,expectedMat);

  cvReleaseMat(&training   );
  cvReleaseMat(&response   );
  cvReleaseMat(&testing    );
  cvReleaseMat(&expected   );
  cvReleaseMat(&responseMat);
  cvReleaseMat(&expectedMat);

  return 0;
}

void cvPrepareResponse(CvMat * response, CvMat * responseMat)
{
  CV_FUNCNAME("cvPrepareResponse");
  int nsamples = response->rows;
  int nclasses = responseMat->cols;
  CV_ASSERT(responseMat->rows==nsamples && response->cols==1 && 
            CV_MAT_TYPE(responseMat->type)==CV_32F);
  __BEGIN__;
  cvZero(responseMat);
  for (int ii=0;ii<nsamples;ii++){
    int label = 0;
    if (CV_MAT_TYPE(response->type)==CV_8U){label=CV_MAT_ELEM(*response,uchar,ii,0);}
    else if (CV_MAT_TYPE(response->type)==CV_32S){label=CV_MAT_ELEM(*response,int,ii,0);}
    else if (CV_MAT_TYPE(response->type)==CV_32F){label=CV_MAT_ELEM(*response,float,ii,0);}
    else {CV_ERROR(CV_StsBadArg,"");}
    CV_MAT_ELEM(*responseMat,float,ii,label)=1;
  }
  fprintf(stderr,"samples:\n");cvPrintf(stderr,"%.0f ",responseMat,cvRect(0,0,nclasses,10));
  __END__;
}


