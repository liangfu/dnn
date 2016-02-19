#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#include "ConvNN.h"

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

int main(int argc, char * argv[])
{
  // if (argc<2){fprintf(stderr,"Error: input training data is required!\n");return 1;}

  const char * training_filename = "../data/mnist/train-images-idx3-ubyte";
  const char * response_filename = "../data/mnist/train-labels-idx1-ubyte";
  const char * testing_filename = "../data/mnist/t10k-images-idx3-ubyte";
  const char * expected_filename = "../data/mnist/t10k-labels-idx1-ubyte";

  const char * pretrained_filename = "../data/mnist/pretrained.xml";

  fprintf(stderr,"Loading MNIST Images ...\n");
  CvMat * training = read_Mnist_Images((char*)training_filename);
  CvMat * response = read_Mnist_Labels((char*)response_filename);
  CvMat * testing = read_Mnist_Images((char*)testing_filename);
  CvMat * expected = read_Mnist_Labels((char*)expected_filename);
  
  int nr = sqrt(training->cols);
  int nc = nr;
  assert(training->cols==nr*nc);
  fprintf(stderr,"%d Images in %dx%d Loaded!\n",training->rows,nr,nc);

  // cvPrintf(stderr,"%d,",response,cvRect(0,0,1,10));
  // {
  // CvMat * sample = cvCreateMat(nr,nc,CV_32F);
  // memcpy(sample->data.ptr,training->data.ptr,sizeof(float)*nr*nc);
  // cvShowImageEx("Test",sample);
  // cvWaitKey();
  // cvReleaseMat(&sample);
  // }

  ConvNN * cnn = new ConvNN(28,28,84,10,0.05/*alpha*/,200/*maxiter*/);
  cnn->createCNN();
  cnn->trainNN(training,response,testing,expected);
  cnn->writeCNNParams(pretrained_filename);

  CNNIO * cnnio = new CNNIO();
  cnnio->init(3,1,1,cnn);
  
  CvMat * result = cvCreateMat(10,1,CV_32F);
  CvMat * sorted = cvCreateMat(result->rows,result->cols,CV_32F);
  CvMat * indices = cvCreateMat(result->rows,result->cols,CV_32S);
  int testCount = 100;int top1=0,top3=0;
  for (int i=0;i<testCount;i++){
    CvMat testing_stub;
    cvGetSubRect(testing,&testing_stub,cvRect(0,i,nr*nc,1));
    cnn->m_cnn->predict(cnn->m_cnn,&testing_stub,result);
    cvSort(result,sorted,indices,CV_SORT_DESCENDING|CV_SORT_EVERY_COLUMN);
    int t1=indices->data.i[0],t2=indices->data.i[1],t3=indices->data.i[2];
    int ex1 = expected->data.ptr[i];
    fprintf(stderr,"label: [%d,%d,%d], expect: %d\n",t1,t2,t3,ex1);
    if (t1==ex1){top1++;}
    if (t1==ex1 || t2==ex1 || t3==ex1){top3++;}
  }
  fprintf(stderr,"top-1: %.0f%%, top-3: %.0f%%\n",
          float(top1*100.f)/float(testCount),
          float(top3*100.f)/float(testCount));
  cvReleaseMat(&result);
  cvReleaseMat(&sorted);
  cvReleaseMat(&indices);

  cvReleaseMat(&training);

  return 0;
}
