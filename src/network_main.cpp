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
  char keys[1<<12];
  sprintf(keys,
          "{  1 |         | train | choose `train` or `test`     }"
          "{  w | weights |       | location of weights file     }"
          "{  m | model   |       | location of model file       }"
          "{  s | solver  |       | location of solver file      }"
          "{  o | omp     | %d    | number of threads to be used }"
          "{  h | help    | false | display this help message    }", 
#ifdef _OPENMP
          size_t(MAX(1.,std::ceil(float(omp_get_max_threads())*.5)))
#else
          1
#endif
          );
  CvCommandLineParser parser(argc,argv,keys);
  const char * task = parser.get<string>("1").c_str();
  const int display_help = parser.get<bool>("help");
  const int max_threads = parser.get<int>("omp");
  if (display_help){parser.printParams();return 0;}
  if (strcmp(task,"train")&&strcmp(task,"test")){
    fprintf(stderr,"choose `train` or `test` as first argument.\n");return 0;
  }
  
  fprintf(stderr, "MAX_THREADS=%d\n",max_threads);

  char   model_filename[1<<10]={0,}; //= parser.get<string>("model").c_str();
  char  solver_filename[1<<10]={0,}; //= parser.get<string>("solver").c_str();
  char weights_filename[1<<10]={0,}; //= parser.get<string>("weights").c_str();
  if (parser.get<string>("model").length()>0){
    strcpy(model_filename,parser.get<string>("model").c_str());
  }
  if (parser.get<string>("solver").length()>0){
    strcpy(solver_filename,parser.get<string>("solver").c_str());
  }
  if (parser.get<string>("weights").length()>0){
    strcpy(weights_filename,parser.get<string>("weights").c_str());
  }

  CvNetwork * cnn = new CvNetwork();
  cnn->loadSolver(solver_filename);

  const char * training_filename = cnn->solver()->training_filename();
  const char * response_filename = cnn->solver()->response_filename();
  const char * testing_filename  = cnn->solver()->testing_filename();
  const char * expected_filename = cnn->solver()->expected_filename();

  fprintf(stderr,"Loading MNIST Images ...\n");
  CvMat * response = (CvMat*)cvLoad((char*)response_filename);
  CvMat * training = (CvMat*)cvLoad((char*)training_filename);
  CvMat * expected = (CvMat*)cvLoad((char*)expected_filename);
  CvMat * testing  = (CvMat*)cvLoad((char*)testing_filename);

  if (!response || !training || !expected || !testing){
    LOGE("Error: not all training/testing files available, try transfer data first.\n"); 
    exit(1);
  }
  
  assert(CV_MAT_TYPE(training->type)==CV_32F);
  assert(training->rows==response->rows);
  
  fprintf(stderr,"%d Training Images Loaded!\n",training->rows);
  fprintf(stderr,"%d Testing Images Loaded!\n",testing->rows);

  CV_TIMER_START();

  if (!strcmp(task,"train")){
    cnn->loadModel(cnn->solver()->model_filename());
    cnn->train(training,response);
    cnn->saveWeights(cnn->solver()->weights_filename());
  }else{
    cnn->loadModel(model_filename);
    cnn->loadWeights(weights_filename);
  }
  
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
