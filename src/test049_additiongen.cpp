#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

void icvConvertIntToBinary(CvMat * trainingInt, CvMat * training);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  s | solver     |       | location of solver file      }"
          "{ tr | trainsize  | 50000 | number of training samples   }"
          "{ ts | testsize   | 10000 | number of testing samples    }"
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

  CvRNG rng;
  CvMat * trainingInt = cvCreateMat(n_train_samples,2,CV_32F);
  CvMat * responseInt = cvCreateMat(n_train_samples,1,CV_32F);
  CvMat * testingInt = cvCreateMat(n_test_samples,2,CV_32F);
  CvMat * expectedInt = cvCreateMat(n_test_samples,1,CV_32F);
  CvMat * training = cvCreateMat(n_train_samples,20,CV_32F);
  CvMat * response = cvCreateMat(n_train_samples,10,CV_32F);
  CvMat * testing = cvCreateMat(n_test_samples,20,CV_32F);
  CvMat * expected = cvCreateMat(n_test_samples,10,CV_32F);

  cvRandArr(&rng,trainingInt,CV_RAND_UNI,cvScalar(10),cvScalar(100));
  cvRandArr(&rng,testingInt,CV_RAND_UNI,cvScalar(10),cvScalar(100));

  CvMat firstcol,secondcol;
  cvGetCol(trainingInt,&firstcol,0);
  cvGetCol(trainingInt,&secondcol,1);
  cvAdd(&firstcol,&secondcol,responseInt);
  cvGetCol(testingInt,&firstcol,0);
  cvGetCol(testingInt,&secondcol,1);
  cvAdd(&firstcol,&secondcol,expectedInt);

  icvConvertIntToBinary(trainingInt,training);
  icvConvertIntToBinary(responseInt,response);
  icvConvertIntToBinary(testingInt,testing);
  icvConvertIntToBinary(expectedInt,expected);

  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,response);
  cvSave(testing_filename_xml,testing);
  cvSave(expected_filename_xml,expected);

  return 0;
}

void icvConvertIntToBinary(CvMat * src, CvMat * dst)
{
  const int nsamples = src->rows;
  const int ndigits = src->cols;
  const int nbinaries = dst->cols/ndigits;
  int number = 0; float * binary = new float[nbinaries];
  assert(dst->rows==nsamples);
  assert(CV_MAT_TYPE(src->type)==CV_32F);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  for (int ii=0;ii<nsamples;ii++){
  for (int jj=0;jj<ndigits;jj++){
    number = CV_MAT_ELEM(*src,float,ii,jj);
    for (int kk=0;kk<nbinaries;kk++){
      binary[kk]=float(int((number>>kk)&0x01)); // reverse order
    }
    memcpy(dst->data.fl+ndigits*nbinaries*ii+jj,binary,sizeof(float)*nbinaries);
  }
  }
  delete [] binary;
}
