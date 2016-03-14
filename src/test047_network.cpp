#include "ml.h"
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
  const char * keys = 
    "{  1 |         | train | choose `train` or `test`     }"
    "{  w | weights |       | location of weights file     }"
    "{  m | model   |       | location of model file       }"
    "{  s | solver  |       | location of solver file      }"
    "{  h | help    | true  | display this help message    }";
    // "{  o | omp     | 1     | number of threads to be used }"
  CvCommandLineParser parser(argc,argv,keys);
  const char * command = parser.get<string>("1").c_str();
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
#ifdef _OPENMP
  int max_threads = MAX(1.,std::ceil(float(omp_get_max_threads())*.5));
#else
  int max_threads = 1;
#endif
  
  fprintf(stderr, "MAX_THREADS=%d\n",max_threads);

  const char * training_filename = "../data/svhn/train/%d.png";
  const char * response_filename = "../data/svhn/train/digitStruct.xml";
  const char * testing_filename  = "../data/svhn/test/%d.png";
  const char * expected_filename = "../data/svhn/test/digitStruct.xml";

  const char * model_filename = "../data/svhn/model.xml";
  const char * solver_filename = "../data/svhn/solver.xml";
  const char * weights_filename = "../data/svhn/weights.xml";

  fprintf(stderr,"Loading MNIST Images ...\n");
  CvMat * response = (CvMat*)cvLoad((char*)response_filename);
  CvMat * training = (CvMat*)cvLoad((char*)training_filename);
  CvMat * expected = (CvMat*)cvLoad((char*)expected_filename);
  CvMat * testing  = (CvMat*)cvLoad((char*)testing_filename);
  
  assert(CV_MAT_TYPE(training->type)==CV_32F);
  assert(training->rows==response->rows);
  
  fprintf(stderr,"%d Training Images Loaded!\n",training->rows);
  fprintf(stderr,"%d Testing Images Loaded!\n",testing->rows);

  CvNetwork * cnn = new CvNetwork();
  cnn->loadModel(model_filename);
  cnn->loadSolver(solver_filename);
  
CV_TIMER_START();
#if 1
  cnn->train(training,response);
  cnn->saveWeights(weights_filename);
#else
  cnn->loadWeights(weights_filename);
#endif
  
  int nsamples = MIN(5000,testing->rows);
  float top1 = cnn->evaluate(testing,expected,nsamples);
  fprintf(stderr,"top-1: %.1f%%\n",float(top1*100.f)/float(nsamples));
CV_TIMER_SHOW();

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&testing);
  cvReleaseMat(&expected);

  return 0;
}
