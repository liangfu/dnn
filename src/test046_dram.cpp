#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "dram.h"

CvMat * icvReadSVHNImages(char * filename)
{
  static const int max_samples = 60000;
  static const int max_strlen = 1000;
  CvMat * data = cvCreateMat(max_samples,max_strlen,CV_8S);
  char * datastr = (char*)malloc(max_strlen);
  int ii;
  for (ii=0;;ii++){
    sprintf(datastr,filename,ii+1);
    FILE * fp = fopen(datastr,"r");
    if (!fp || (ii+1)==max_samples){break;}
    memcpy(data->data.ptr+max_strlen*ii,datastr,max_strlen);
    fclose(fp);
  }
  data->rows = ii;
  free(datastr);
  return data;
}

CvMat * icvReadSVHNLabels(char * filename)
{
  CvFileStorage * fs = cvOpenFileStorage(filename,0,CV_STORAGE_READ);
  if (!fs){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  CvFileNode * fnode = cvGetRootFileNode(fs);
  char tagname[20]; int ii;
  static const int max_samples = 60000;
  static const int nparams = 4;
  CvMat * data = cvCreateMat(max_samples,1+nparams*10,CV_32S);
  for (ii=0;;ii++){
    sprintf(tagname,"img%d",ii+1);
    CvMat * sample = (CvMat*)cvReadByName(fs,fnode,tagname);
    if (!sample || (ii+1)==max_samples){break;}
    int nnumbers = sample->rows;
    CV_MAT_ELEM(*data,int,ii,0)=nnumbers;
    for (int jj=0;jj<nnumbers;jj++){
      float xx = CV_MAT_ELEM(*sample,float,jj,0);
      float yy = CV_MAT_ELEM(*sample,float,jj,1);
      float ww = CV_MAT_ELEM(*sample,float,jj,2);
      float hh = CV_MAT_ELEM(*sample,float,jj,3);
      float ll = CV_MAT_ELEM(*sample,float,jj,4);
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+0)=cvRound((xx+ww)*.5f);  // x
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+1)=cvRound((yy+hh)*.5f);  // y
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+2)=cvRound(MAX(ww,hh)); // scale
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+3)=cvRound(ll);           // label
    }
    cvReleaseMat(&sample);
  }
  data->rows = ii;
  cvReleaseFileStorage(&fs);
  return data;
}

int main(int argc, char * argv[])
{
  // if (argc<2){fprintf(stderr,"Error: input training data is required!\n");return 1;}
#ifdef _OPENMP
  const int max_threads = MAX(1.,std::ceil(float(omp_get_max_threads())*.5));
#else
  const int max_threads = 1;
#endif
  fprintf(stderr, "MAX_THREADS=%d\n",max_threads);

  const char * training_filename = "../data/svhn/train/%d.png";
  const char * response_filename = "../data/svhn/train/digitStruct.xml";
  const char * testing_filename  = "../data/svhn/test/%d.png";
  const char * expected_filename = "../data/svhn/test/digitStruct.xml";
  const char * pretrained_filename = "../data/svhn/pretrained.xml";

  fprintf(stderr,"Loading MNIST Images ...\n");
  CvMat * training = icvReadSVHNImages((char*)training_filename);
  CvMat * response = icvReadSVHNLabels((char*)response_filename);
  CvMat * testing  = icvReadSVHNImages((char*)testing_filename);
  CvMat * expected = icvReadSVHNLabels((char*)expected_filename);

  assert(training->rows==response->rows);
  assert(testing->rows==expected->rows);
  
  fprintf(stderr,"%d Training Images Loaded!\n",training->rows);
  fprintf(stderr,"%d Testing Images Loaded!\n",testing->rows);

  DRAM * cnn = new DRAM(28,28, // input image size
                        84,10, // full connect nodes
                        0.05,  // learning rate
                        1000,  // maxiter
                        2      // batch_size
                        );
  cnn->createNetwork();
CV_TIMER_START();
#if 1
  cnn->trainNetwork(training,response,testing,expected);
  cnn->writeNetworkParams(pretrained_filename);
#else
  cnn->readNetworkParams(pretrained_filename);
#endif

  // CNNIO * cnnio = new CNNIO();
  // cnnio->init(3,1,1,cnn);
  
  int nsamples = MIN(5000,testing->rows);
  CvMat * samples = cvCreateMat(nsamples,testing->cols,CV_32F);
  CvMat * result = cvCreateMat(10,nsamples,CV_32F);
  CvMat * sorted = cvCreateMat(result->rows,result->cols,CV_32F);
  CvMat * indices = cvCreateMat(result->rows,result->cols,CV_32S);
  CvMat * indtop1 = cvCreateMat(1,result->cols,CV_32S);
  CvMat * expected_submat = cvCreateMat(nsamples,1,CV_8U);
  CvMat * expected_converted = cvCreateMat(nsamples,1,CV_32S);
  CvMat * expected_transposed = cvCreateMat(1,result->cols,CV_32S);
  CvMat * indtop1res = cvCreateMat(1,result->cols,CV_8U);
  // testing data
  cvGetRows(testing,samples,0,nsamples);
  cnn->m_cnn->predict(cnn->m_cnn,samples,result);
  cvSort(result,sorted,indices,CV_SORT_DESCENDING|CV_SORT_EVERY_COLUMN);
  cvGetRow(indices,indtop1,0);
  // expected data
  cvGetRows(expected,expected_submat,0,nsamples);
  cvConvert(expected_submat,expected_converted);
  cvTranspose(expected_converted,expected_transposed);
  cvCmp(indtop1,expected_transposed,indtop1res,CV_CMP_EQ);
#if 0
  fprintf(stderr,"expected:\n\t");
  cvPrintf(stderr,"%d,",expected_transposed);
  fprintf(stderr,"result:\n\t");
  cvPrintf(stderr,"%d,",indtop1);
#endif
  float top1=cvSum(indtop1res).val[0]/255.f;
  cvReleaseMat(&samples);
  cvReleaseMat(&result);
  cvReleaseMat(&sorted);
  cvReleaseMat(&indices);
  cvReleaseMat(&indtop1);
  cvReleaseMat(&expected_submat);
  cvReleaseMat(&expected_converted);
  cvReleaseMat(&expected_transposed);
  cvReleaseMat(&indtop1res);

  fprintf(stderr,"top-1: %.1f%%\n",float(top1*100.f)/float(nsamples));
CV_TIMER_SHOW();

  cvReleaseMat(&training);

  return 0;
}
