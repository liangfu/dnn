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

void cvGenerateMultiDigitMNIST(CvMat * training, CvMat * training_multi, 
                               CvMat * response, CvMat * response_multi, const int ndigits);

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
  const int testsize = parser.get<int>("testsize");
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
  const int ndigits = 3;
  CvMat * training_multi = cvCreateMat(trainsize,imsize*imsize,CV_32F);
  CvMat * response_multi = cvCreateMat(trainsize,ndigits+10*ndigits,CV_32F);
  CvMat *  testing_multi = cvCreateMat( testsize,imsize*imsize,CV_32F);
  CvMat * expected_multi = cvCreateMat( testsize,ndigits+10*ndigits,CV_32F);

  cvGenerateMultiDigitMNIST(training, training_multi, response, response_multi, ndigits);
  cvGenerateMultiDigitMNIST( testing,  testing_multi, expected, expected_multi, ndigits);

  cvSave(training_filename_xml,training_multi);
  cvSave(response_filename_xml,response_multi);
  cvSave( testing_filename_xml, testing_multi);
  cvSave(expected_filename_xml,expected_multi);

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&testing );
  cvReleaseMat(&expected);
  cvReleaseMat(&training_multi);
  cvReleaseMat(&response_multi);
  cvReleaseMat( &testing_multi);
  cvReleaseMat(&expected_multi);

  return 0;
}

void cvGenerateMultiDigitMNIST(CvMat * training, CvMat * training_multi, 
                               CvMat * response, CvMat * response_multi, const int ndigits)
{
  CV_FUNCNAME("cvGenerateMultiDigitMNIST");
  __BEGIN__;
  CV_ASSERT(training->cols==28*28 && training_multi->cols==64*64);
  // CV_ASSERT(training->rows==training_multi->rows);
  CV_ASSERT(CV_MAT_TYPE(training_multi->type)==CV_32F);
  CV_ASSERT(CV_MAT_TYPE(training->type)==CV_32F);
  CV_ASSERT(CV_MAT_TYPE(response->type)==CV_8U);
  CV_ASSERT(CV_MAT_TYPE(response_multi->type)==CV_32F);
  CV_ASSERT(response_multi->cols==(ndigits+10*ndigits));
  int imsize = sqrt(training_multi->cols);
  CvMat * sample = cvCreateMat(28,28,CV_32F);
  CvMat * target0 = cvCreateMat(imsize,imsize,CV_32F);
  CvMat * target = cvCreateMat(imsize,imsize,CV_32F);
  CvMat * warp_p = cvCreateMat(2,3,CV_32F);
  CvRNG rng = cvRNG(-1); int tidx = 0, nd = 0;
  CvMat * vmat = cvCreateMat(1,ndigits,CV_32S); cvZero(vmat); int * vptr = vmat->data.i;
  cvZero(response_multi);
  for (int idx=0;idx<training_multi->rows;idx++){
    cvZero(target); cvSet(vmat,cvScalar(-1));

    nd = (cvRandInt(&rng)%ndigits)+1; // number of digits in current frame
    CV_MAT_ELEM(*response_multi,float,idx,nd-1)=1;

    for (int diter=0;diter<nd;diter++){
      tidx = cvRandInt(&rng) % training->rows;
      vptr[diter] = CV_MAT_ELEM(*response,uchar,tidx,0);
      memcpy(sample->data.ptr,training->data.ptr+training->step*tidx,training->step);
      warp_p->data.fl[0]=warp_p->data.fl[4]=1;
      warp_p->data.fl[2]=-cvRandReal(&rng)*2-20+9*(nd-1)-18*diter;
      warp_p->data.fl[5]=-cvRandReal(&rng)*6-10; // cvPrintf(stderr,"%.1f ",warp_p);
      icvWarp(sample,target0,warp_p);
      cvAdd(target0, target, target); 
      CV_MAT_ELEM(*response_multi,float,idx,ndigits+10*diter+vptr[diter])=1;
    }
    for (int diter=nd;diter<ndigits;diter++){
    for (int ii=0;ii<10;ii++){
      CV_MAT_ELEM(*response_multi,float,idx,ndigits+10*diter+ii)=.1f;
    }
    }
    cvMinS(target,255,target);
    memcpy(training_multi->data.ptr+training_multi->step*idx,target->data.ptr,training_multi->step);

    // visualize
    CvMat response_submat_hdr;
    // cvGetRow(response_multi,&response_submat_hdr,idx);
    // cvPrintf(stderr,"%.1f ", &response_submat_hdr);
    // cvPrintf(stderr,"%d ",vmat);
    // CV_SHOW(target);
  }
  cvReleaseMat(&vmat);
  cvReleaseMat(&sample);
  cvReleaseMat(&target0);
  cvReleaseMat(&target);
  cvReleaseMat(&warp_p);
  __END__;
}
