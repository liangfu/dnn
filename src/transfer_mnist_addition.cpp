#include "dnn.h"
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

void cvGenerateAdditionMNIST(CvMat * training, CvMat * training_twodigit, 
                             CvMat * response, CvMat * response_twodigit);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  s | solver  |       | location of solver file      }"
          "{ tr | trainsize  | 5000  | number of training samples   }"
          "{ ts | testsize   | 1000  | number of testing samples    }"
          "{  h | help    | false | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const int display_help = parser.get<bool>("help");
  const int trainsize = parser.get<int>("trainsize");
  const int  testsize = parser.get<int>("testsize");
  if (display_help){parser.printParams();return 0;}
  const char * solver_filename  = parser.get<string>("solver").c_str();
  Network * cnn = new Network();
  cnn->loadSolver(solver_filename);
  const char * training_filename_xml = cnn->solver()->training_filename();
  const char * response_filename_xml = cnn->solver()->response_filename();
  const char *  testing_filename_xml  = cnn->solver()->testing_filename();
  const char * expected_filename_xml = cnn->solver()->expected_filename();
  const char * training_filename = "data/mnist/train-images-idx3-ubyte"; 
  const char * response_filename = "data/mnist/train-labels-idx1-ubyte"; 
  const char *  testing_filename = "data/mnist/t10k-images-idx3-ubyte"; 
  const char * expected_filename = "data/mnist/t10k-labels-idx1-ubyte"; 

  const int nclasses = 10;
  CvMat * training    = read_Mnist_Images((char*)training_filename);
  CvMat * response    = read_Mnist_Labels((char*)response_filename);
  CvMat * testing     = read_Mnist_Images((char*)testing_filename);
  CvMat * expected    = read_Mnist_Labels((char*)expected_filename);
  assert(training->rows==response->rows && testing->rows==expected->rows);

  const int imsize = 64;
  CvMat * training_twodigit = cvCreateMat(trainsize,imsize*imsize,CV_32F);
  CvMat * response_twodigit = cvCreateMat(trainsize,19,CV_32F);
  CvMat *  testing_twodigit = cvCreateMat( testsize,imsize*imsize,CV_32F);
  CvMat * expected_twodigit = cvCreateMat( testsize,19,CV_32F);

  cvGenerateAdditionMNIST(training, training_twodigit, response, response_twodigit);
  cvGenerateAdditionMNIST( testing,  testing_twodigit, expected, expected_twodigit);

  cvSave(training_filename_xml,training_twodigit);
  cvSave(response_filename_xml,response_twodigit);
  cvSave( testing_filename_xml, testing_twodigit);
  cvSave(expected_filename_xml,expected_twodigit);

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&testing );
  cvReleaseMat(&expected);
  cvReleaseMat(&training_twodigit);
  cvReleaseMat(&response_twodigit);
  cvReleaseMat( &testing_twodigit);
  cvReleaseMat(&expected_twodigit);

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

void cvGenerateAdditionMNIST(CvMat * training, CvMat * training_twodigit, 
                             CvMat * response, CvMat * response_twodigit)
{
  CV_FUNCNAME("cvGenerateTwoDigitMNIST");
  __BEGIN__;
  CV_ASSERT(training->cols==28*28 && training_twodigit->cols==64*64);
  // CV_ASSERT(training->rows==training_twodigit->rows);
  CV_ASSERT(CV_MAT_TYPE(training_twodigit->type)==CV_32F);
  CV_ASSERT(CV_MAT_TYPE(response->type)==CV_8U);
  CV_ASSERT(CV_MAT_TYPE(response_twodigit->type)==CV_32F);
  CV_ASSERT(response_twodigit->cols==19);
  int imsize = sqrt(training_twodigit->cols);
  CvMat * sample = cvCreateMat(28,28,CV_32F);
  CvMat * target0 = cvCreateMat(imsize,imsize,CV_32F);
  CvMat * target = cvCreateMat(imsize,imsize,CV_32F);
  CvMat * warp_p = cvCreateMat(2,3,CV_32F);
  CvRNG rng = cvRNG(-1); int tidx = 0; int vals[2] = {0,0};
  CvMat vmat = cvMat(1,2,CV_32S,vals);
  cvZero(response_twodigit);
  for (int idx=0;idx<training_twodigit->rows;idx++){
    cvZero(target);
    tidx = cvFloor(cvRandReal(&rng)*training->rows);
    vals[0] = CV_MAT_ELEM(*response,uchar,tidx,0);
    memcpy(sample->data.ptr,training->data.ptr+training->step*tidx,training->step);
    warp_p->data.fl[0]=warp_p->data.fl[4]=1;
    warp_p->data.fl[2]=-cvRandReal(&rng)*(32-28);
    warp_p->data.fl[5]=-cvRandReal(&rng)*(64-28); // cvPrintf(stderr,"%f ",warp_p);
    icvWarp(sample,target0,warp_p);
    tidx = cvFloor(cvRandReal(&rng)*training->rows);
    vals[1] = CV_MAT_ELEM(*response,uchar,tidx,0);
    memcpy(sample->data.ptr,training->data.ptr+training->step*tidx,training->step);
    warp_p->data.fl[0]=warp_p->data.fl[4]=1;
    warp_p->data.fl[2]=-(cvRandReal(&rng)*(32-28))-32;
    warp_p->data.fl[5]=-cvRandReal(&rng)*(64-28); // cvPrintf(stderr,"%f ",warp_p);
    icvWarp(sample,target,warp_p); 
    cvAdd(target0, target, target); 
    CV_MAT_ELEM(*response_twodigit,float,idx,vals[0]+vals[1])=1;
    memcpy(target->data.ptr,training_twodigit->data.ptr+training_twodigit->step*idx,training_twodigit->step);

    // visualize
    // cvPrintf(stderr,"%d ",&vmat);
    // CV_SHOW(target);
  }
  cvReleaseMat(&sample);
  cvReleaseMat(&target0);
  cvReleaseMat(&target);
  cvReleaseMat(&warp_p);
  __END__;
}
