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
CvMat * icvReadPrimateImages(char * filename, CvMat * response);
CvMat * icvReadPrimateLabels(char * filename, const int max_samples);
CvMat * icvConvertLabels(CvMat * src);

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
  const char * training_filename = "data/primate/train/%d.png";
  const char * response_filename = "data/primate/train/digitStruct.xml";
  const char *  testing_filename = "data/primate/test/%d.png";
  const char * expected_filename = "data/primate/test/digitStruct.xml";
  const char * training_filename_xml = cnn->solver()->training_filename();
  const char * response_filename_xml = cnn->solver()->response_filename();
  const char *  testing_filename_xml = cnn->solver()->testing_filename();
  const char * expected_filename_xml = cnn->solver()->expected_filename();
  const int n_train_samples = parser.get<int>("trainsize");
  const int n_test_samples = parser.get<int>("testsize");
  const int ndigits = 3;

  fprintf(stderr,"Loading Primate Images ...\n");
  CvMat * response = icvReadPrimateLabels((char*)response_filename,n_train_samples);
  CvMat * training = icvReadPrimateImages((char*)training_filename,response);
  assert(CV_MAT_TYPE(training->type)==CV_32F);
  CvMat * expected = icvReadPrimateLabels((char*)expected_filename,n_test_samples);
  CvMat * testing  = icvReadPrimateImages((char*) testing_filename,expected);
  CvMat * response_mat = icvConvertLabels(response);
  CvMat * expected_mat = icvConvertLabels(expected);

  fprintf(stderr,"%d training samples generated!\n", training->rows);
  fprintf(stderr,"%d testing samples generated!\n", testing->rows);

  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,response_mat);
  cvSave( testing_filename_xml,testing);
  cvSave(expected_filename_xml,expected_mat);

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&response_mat);
  cvReleaseMat(&testing);
  cvReleaseMat(&expected);
  cvReleaseMat(&expected_mat);
  
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

CvMat * icvReadPrimateImages(char * filename, CvMat * response)
{
  const int max_samples = response->rows;
  const int max_strlen = 64*64;
  CvMat * data = cvCreateMat(max_samples,max_strlen,CV_32F);
  char datastr[1000];
  int ii; CvMat hdr;
  CvMat * sample = cvCreateMat(64,64,CV_32F);
  CvMat * warp = cvCreateMat(2,3,CV_32F);
  for (ii=0;;ii++){
    sprintf(datastr,filename,ii+1);
    IplImage * img = cvLoadImage(datastr, CV_LOAD_IMAGE_GRAYSCALE);
    if (!img || ii==max_samples){break;}
    CvMat * mat = cvCreateMat(img->height,img->width,CV_32F);
    cvConvert(img,mat);
    int nimages = CV_MAT_ELEM(*response,int,ii,0);
    float ww = CV_MAT_ELEM(*response,int,ii,3);
    float hh = CV_MAT_ELEM(*response,int,ii,4);
    float xx = CV_MAT_ELEM(*response,int,ii,1)-ww*.5f;
    float yy = CV_MAT_ELEM(*response,int,ii,2)-hh*.5f;
    for (int jj=1;jj<nimages;jj++){
      float www = CV_MAT_ELEM(*response,int,ii,1+4*jj+2);
      float hhh = CV_MAT_ELEM(*response,int,ii,1+4*jj+3);
      float xxx = CV_MAT_ELEM(*response,int,ii,1+4*jj+0)-www*.5f;
      float yyy = CV_MAT_ELEM(*response,int,ii,1+4*jj+1)-hhh*.5f;
      xx = MIN(xx,xxx); yy = MIN(yy,yyy);
      ww = MAX(xxx+www-xx,ww);
      hh = MAX(yyy+hhh-yy,hh);
    }
    xx+=ww*.5f;yy+=hh*.5f; float ss = MAX(ww,hh);
    cvZero(warp);cvZero(sample);
    warp->data.fl[2]=xx-ss*.5f-ss*.15f;
    warp->data.fl[5]=yy-ss*.5f-ss*.15f;
    warp->data.fl[0]=warp->data.fl[4]=ss*1.3f/64.f; //
    icvWarp(mat,sample,warp);
    CvScalar avg,sdv;
    cvAvgSdv(sample,&avg,&sdv);
    cvSubS(sample,avg,sample);
    cvScale(sample,sample,1.f/sdv.val[0]);
#if 0 // debug
    cvPrintf(stderr,"%f,",warp);
    cvAvgSdv(sample,&avg,&sdv); // re-calculate
    fprintf(stderr,"avg: %f, sdv: %f\n--\n",avg.val[0],sdv.val[0]);
    CV_SHOW(sample); // CV_SHOW(mat);
#endif
    memcpy(data->data.fl+max_strlen*ii,sample->data.ptr,max_strlen*sizeof(float));
    cvReleaseImage(&img);
    cvReleaseMat(&mat);
  }
  data->rows = ii;
  cvReleaseMat(&sample);
  cvReleaseMat(&warp);
  assert(CV_MAT_TYPE(data->type)==CV_32F);
  return data;
}

CvMat * icvReadPrimateLabels(char * filename, const int max_samples)
{
  CvFileStorage * fs = cvOpenFileStorage(filename,0,CV_STORAGE_READ);
  if (!fs){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  CvFileNode * fnode = cvGetRootFileNode(fs);
  char tagname[20]; int ii;
  const int nparams = 4;
  CvMat * data = cvCreateMat(max_samples,1+nparams*10,CV_32S); cvZero(data);
  for (ii=0;;ii++){
    sprintf(tagname,"img%d",ii+1);
    CvMat * sample = (CvMat*)cvReadByName(fs,fnode,tagname);
    if (!sample || ii==max_samples){break;}
    int nnumbers = sample->rows;
    CV_MAT_ELEM(*data,int,ii,0)=nnumbers;
    for (int jj=0;jj<nnumbers;jj++){
      float xx = CV_MAT_ELEM(*sample,float,jj,0);
      float yy = CV_MAT_ELEM(*sample,float,jj,1);
      float ww = CV_MAT_ELEM(*sample,float,jj,2);
      float hh = CV_MAT_ELEM(*sample,float,jj,3);
      float ll = CV_MAT_ELEM(*sample,float,jj,4);
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+0)=cvRound(xx+ww*.5f);  // x
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+1)=cvRound(yy+hh*.5f);  // y
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+2)=cvRound(MAX(ww,hh));   // scale
      CV_MAT_ELEM(*data,int,ii,1+nparams*jj+3)=cvRound(ll);           // label
    }
    cvReleaseMat(&sample);
  }
  data->rows = ii;
  cvReleaseFileStorage(&fs);
  return data;
}

CvMat * icvConvertLabels(CvMat * src)
{
  CV_FUNCNAME("icvConvertLabels");
  CvMat * dst = 0;
  __BEGIN__;
  const int nsamples = src->rows;
  const int ntargets = 5;
  const int nclasses = 10; // decimal
  const int nparams = 4;
  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_32S);
  dst = cvCreateMat(nsamples,(ntargets+1)*nclasses,CV_32F); cvZero(dst);
  for (int ii=0;ii<nsamples;ii++){
  int nlabels = CV_MAT_ELEM(*src,int,ii,0);
  CV_MAT_ELEM(*dst,float,ii,nlabels)=1;
  for (int jj=0;jj<nlabels;jj++){
    int label = CV_MAT_ELEM(*src,int,ii,1+nparams*jj+3); // label
    CV_MAT_ELEM(*dst,float,ii,10+nclasses*jj+label)=1;
  }
  }
  __END__;
  return dst;
}
