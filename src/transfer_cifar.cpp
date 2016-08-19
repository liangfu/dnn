#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "highgui.h"
#include "cxcore.h"
#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;
typedef std::string string;

void cvConvertIntToDecimal(const int ndigits, CvMat * src, CvMat * dst);
void cvReadCIFAR_train(CvMat * training, CvMat * response);
void cvReadCIFAR_test(CvMat * training, CvMat * response);

// #define DEBUG 1

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  1 |       | cifar10 | choose `cifar10` or `cifar100` }"
          "{  s | solver     |       | location of solver file      }"
          "{  h | help  | false   | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const char * task = parser.get<string>("1").c_str();
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
  int n_classes = (!strcmp(task,"cifar10"))?10:100;
  const char * solver_filename  = parser.get<string>("solver").c_str();
  Network * cnn = new Network();
  cnn->loadSolver(solver_filename);

  const char * training_filename_xml = cnn->solver()->training_filename();
  const char * response_filename_xml = cnn->solver()->response_filename();
  const char *  testing_filename_xml = cnn->solver()->testing_filename();
  const char * expected_filename_xml = cnn->solver()->expected_filename();

  fprintf(stderr,"Loading CIFAR Images ...\n");
  CvMat * training = cvCreateMat(50000,3072,CV_8U);
  CvMat * response = cvCreateMat(50000,n_classes,CV_8U);
  cvReadCIFAR_train(training,response);
  CvMat * testing = cvCreateMat(10000,3072,CV_8U);
  CvMat * expected = cvCreateMat(10000,n_classes,CV_8U);
  cvReadCIFAR_test(testing,expected);

  fprintf(stderr,"%d training samples generated!\n", training->rows);
  fprintf(stderr,"%d testing samples generated!\n", testing->rows);

  fprintf(stderr,"Saving CIFAR Images ...\n");
  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,response);
  cvSave( testing_filename_xml,testing);
  cvSave(expected_filename_xml,expected);
  fprintf(stderr,"Done!\n");

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&testing);
  cvReleaseMat(&expected);
  
  return 0;
}

void cvConvertIntToDecimal(const int ndigits, CvMat * src, CvMat * dst)
{
  const int nsamples = src->rows;
  const int nnumbers = src->cols;
  assert(dst->rows==nsamples);
  assert(CV_MAT_TYPE(src->type)==CV_32S);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  CvMat * values = cvCreateMat(ndigits,10,CV_32F);
  int stepsize = ndigits*10*sizeof(float);
  for (int ii=0;ii<nsamples;ii++){
#ifdef DEBUG
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
#ifdef DEBUG
    fprintf(stderr,"\noutput:\n");
    cvPrintf(stderr,"%.0f ",dst,cvRect(0,ii,dst->cols,1));
#endif
  }
  cvReleaseMat(&values);
}

void cvReadCIFAR_train(CvMat * images, CvMat * labels)
{
  CV_FUNCNAME("cvReadCIFAR");
  __CV_BEGIN__;
  CV_ASSERT(CV_MAT_TYPE(images->type)==CV_8U && CV_MAT_TYPE(labels->type)==CV_8U);
  cvZero(images); cvZero(labels);
  const int n_classes=labels->cols;
  char batch_fname[1024];
  uchar label=0;
  for (int batch_index=1; batch_index<=5; batch_index++){
    memset(batch_fname,0,sizeof(batch_fname));
    sprintf(batch_fname,"data/cifar/cifar-%d-batches-bin/data_batch_%d.bin",n_classes,batch_index);
    fprintf(stderr,"Converting `%s`\n",batch_fname);
    FILE * fp = fopen(batch_fname,"rb");
    if (!fp){fprintf(stderr,"ERROR: file `%s` not available to read.\n",batch_fname);exit(-1);}
    const int offset=(batch_index-1)*10000;
    for (int ii=0; ii<10000; ii++){
      fread(&label,1,1,fp); fread(images->data.ptr+offset*3072+3072*ii,1,3072,fp);
      (labels->data.ptr+offset*n_classes+n_classes*ii)[label]=1;
    }
    fclose(fp);
  }
  __CV_END__;
}

void cvReadCIFAR_test(CvMat * images, CvMat * labels)
{
  CV_FUNCNAME("cvReadCIFAR");
  __CV_BEGIN__;
  CV_ASSERT(CV_MAT_TYPE(images->type)==CV_8U && CV_MAT_TYPE(labels->type)==CV_8U);
  cvZero(images); cvZero(labels);
  const int n_classes=labels->cols; uchar label=0;
  char batch_fname[1024]; memset(batch_fname,0,sizeof(batch_fname));
  sprintf(batch_fname,"data/cifar/cifar-%d-batches-bin/test_batch.bin",n_classes);
  FILE * fp = fopen(batch_fname,"rb");
  if (!fp){fprintf(stderr,"ERROR: file `%s` not available to read.\n",batch_fname);exit(-1);}
  for (int ii=0; ii<10000; ii++){
    fread(&label,1,1,fp); fread(images->data.ptr+3072*ii,1,3072,fp);
    (labels->data.ptr+n_classes*ii)[label]=1;
  }
  fclose(fp);
  __CV_END__;
}
