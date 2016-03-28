#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

void icvConvertIntToDecimal(const int ndigits, CvMat * src, CvMat * dst);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  s | solver     |       | location of solver file      }"
          "{ tr | trainsize  | 5000  | number of training samples   }"
          "{ ts | testsize   | 1000  | number of testing samples    }"
          "{  h | help       | false | display this help message    }");
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
  const int n_train_samples = parser.get<int>("trainsize");
  const int n_test_samples = parser.get<int>("testsize");
  const int ndigits = 3;

  CvRNG rng = cvRNG(-1);
  CvMat * trainingInt = cvCreateMat(n_train_samples,2,CV_32S);
  CvMat * responseInt = cvCreateMat(n_train_samples,1,CV_32S);
  CvMat * testingInt  = cvCreateMat(n_test_samples,2,CV_32S);
  CvMat * expectedInt = cvCreateMat(n_test_samples,1,CV_32S);
  CvMat * training = cvCreateMat(n_train_samples,2*ndigits*10,CV_32F);
  CvMat * response = cvCreateMat(n_train_samples,1*ndigits*10,CV_32F);
  CvMat * testing  = cvCreateMat(n_test_samples, 2*ndigits*10,CV_32F);
  CvMat * expected = cvCreateMat(n_test_samples, 1*ndigits*10,CV_32F);

  cvRandArr(&rng,trainingInt,CV_RAND_UNI,cvScalar(10),cvScalar(pow(10.f,ndigits)*.5f));
  cvRandArr(&rng,testingInt, CV_RAND_UNI,cvScalar(10),cvScalar(pow(10.f,ndigits)*.5f));

  CvMat firstcol,secondcol;
  cvGetCol(trainingInt,&firstcol,0);
  cvGetCol(trainingInt,&secondcol,1);
  cvAdd(&firstcol,&secondcol,responseInt);
  cvGetCol(testingInt,&firstcol,0);
  cvGetCol(testingInt,&secondcol,1);
  cvAdd(&firstcol,&secondcol,expectedInt);

  icvConvertIntToDecimal(ndigits,trainingInt,training);
  icvConvertIntToDecimal(ndigits,responseInt,response);
  icvConvertIntToDecimal(ndigits,testingInt,testing);
  icvConvertIntToDecimal(ndigits,expectedInt,expected);

  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,response);
  cvSave(testing_filename_xml,testing);
  cvSave(expected_filename_xml,expected);

  return 0;
}

void icvConvertIntToDecimal(const int ndigits, CvMat * src, CvMat * dst)
{
  const int nsamples = src->rows;
  const int nnumbers = src->cols;
  assert(dst->rows==nsamples);
  assert(CV_MAT_TYPE(src->type)==CV_32S);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  CvMat * values = cvCreateMat(ndigits,10,CV_32F);
  int stepsize = ndigits*10*sizeof(float);
  for (int ii=0;ii<nsamples;ii++){
#if 0 // debug
    fprintf(stderr,"number: ");
    for (int jj=0;jj<nnumbers;jj++){
      fprintf(stderr,"%d ",CV_MAT_ELEM(*src,int,ii,jj));
    }
#endif
    for (int jj=0;jj<nnumbers;jj++){
      cvZero(values);
      int number = CV_MAT_ELEM(*src,int,ii,jj);
      for (int kk=0;kk<ndigits;kk++){
        int pos = cvFloor((number%int(pow(10.f,kk+1)))/pow(10.f,kk));
        CV_MAT_ELEM(*values,float,kk,pos)=1;
      }
      memcpy(dst->data.ptr+stepsize*(nnumbers*ii+jj),values->data.ptr,stepsize);
    }
#if 0 // debug
    fprintf(stderr,"\noutput:\n");
    cvPrintf(stderr,"%.0f ",dst,cvRect(0,ii,dst->cols,1));
#endif
  }
  cvReleaseMat(&values);
}
