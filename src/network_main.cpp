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
          "{  1 |         | train | choose `train` or `test`     }"
          "{  s | solver  |       | location of solver file      }"
          "{  o | omp     | %d    | number of threads to be used }"
          "{  h | help    | false | display this help message    }", 
#ifdef _OPENMP
          int(size_t(MAX(1.,std::ceil(float(omp_get_max_threads())*.5))))
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

  char solver_filename[1<<10]={0,};
  if (parser.get<string>("solver").length()>0){
    strcpy(solver_filename,parser.get<string>("solver").c_str());
  }else{
    LOGE("solver filename is empty."); exit(-1);
  }

  Network * cnn = new Network();
  cnn->loadSolver(solver_filename);
  cnn->loadModel(cnn->solver()->model_filename());

  const char * training_filename = cnn->solver()->training_filename();
  const char * response_filename = cnn->solver()->response_filename();
  const char * testing_filename  = cnn->solver()->testing_filename();
  const char * expected_filename = cnn->solver()->expected_filename();
  const char * predicted_filename = cnn->solver()->predicted_filename();

  fprintf(stderr,"Loading Dataset ...\n");
  
  if (!strcmp(task,"train")){
    CvMat * training = (CvMat*)cvLoad((char*)training_filename);
    CvMat * response = (CvMat*)cvLoad((char*)response_filename);
    if (!response || !training){
      LOGE("error: not all training files available, try transfer data first.\n"); return -1;
    }
    assert(CV_MAT_TYPE(training->type)==CV_32F);
    assert(training->rows==response->rows);
    fprintf(stderr,"%d Training Images Loaded!\n",training->rows);
    CV_TIMER_START();
    cnn->train(training,response);
    cnn->saveWeights(cnn->solver()->weights_filename());
    CV_TIMER_SHOW();
    cvReleaseMat(&training);
    cvReleaseMat(&response);
  }else{
    CvMat * testing  = (CvMat*)cvLoad((char*) testing_filename);
    CvMat * expected = 0;//(CvMat*)cvLoad((char*)expected_filename);
    if (!testing){
      LOGE("error: testing file not available, try transfer data first.\n"); return -1;
    }
    assert(CV_MAT_TYPE(testing->type)==CV_32F);
    if (expected){assert( testing->rows==expected->rows);}
    fprintf(stderr,"%d Testing Images Loaded!\n",testing->rows);
    CV_TIMER_START();
    cnn->loadWeights(cnn->solver()->weights_filename());
#if 0
    cnn->evaluate(testing,expected,testing->rows,predicted_filename);
#else
    cnn->evaluate(testing,expected,5,predicted_filename);
#endif
    CV_TIMER_SHOW();
    if (testing){cvReleaseMat(&testing);testing=0;}
    if (expected){cvReleaseMat(&expected);expected=0;}
  }

  return 0;
}
