#include "dnn.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

CvMat * icvReadPrimateImages(char * filename, const int max_samples);
CvMat * icvReadPrimateLabels(char * filename, const int max_samples);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  s | solver     |       | location of solver file      }"
          "{ tr | trainsize  | 180   | number of training samples   }"
          "{ ts | testsize   | 60    | number of testing samples    }"
          "{  h | help       | false | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
  const char * solver_filename  = parser.get<string>("solver").c_str();
  CvNetwork * cnn = new CvNetwork();
  cnn->loadSolver(solver_filename);
  const char * response_filename = "data/primate/train.yml";
  const char * expected_filename = "data/primate/test.yml";
  const char * training_filename_xml = cnn->solver()->training_filename();
  const char * response_filename_xml = cnn->solver()->response_filename();
  const char *  testing_filename_xml = cnn->solver()->testing_filename();
  const char * expected_filename_xml = cnn->solver()->expected_filename();
  const int trainsize = parser.get<int>("trainsize");
  const int testsize = parser.get<int>("testsize");

  fprintf(stderr,"Loading Primate Images ...\n");
  CvMat * response = icvReadPrimateLabels((char*)response_filename,trainsize);
  CvMat * training = icvReadPrimateImages((char*)response_filename,trainsize);
  assert(CV_MAT_TYPE(training->type)==CV_32F);
  CvMat * expected = icvReadPrimateLabels((char*)expected_filename,testsize);
  CvMat * testing  = icvReadPrimateImages((char*)expected_filename,testsize);

  fprintf(stderr,"%d training samples generated!\n", training->rows);
  fprintf(stderr,"%d testing samples generated!\n", testing->rows);

  cvSave(training_filename_xml,training);
  cvSave(response_filename_xml,response);
  cvSave( testing_filename_xml,testing);
  cvSave(expected_filename_xml,expected);

  cvReleaseMat(&training);
  cvReleaseMat(&response);
  cvReleaseMat(&testing);
  cvReleaseMat(&expected);
  
  return 0;
}

CvMat * icvReadPrimateImages(char * filename, const int max_samples)
{
  CV_FUNCNAME("icvReadPrimateImages");
  static const int imsize = 240*240;
  CvMat * data = cvCreateMat(max_samples,imsize,CV_32F); cvZero(data);
  __BEGIN__;
  CvFileStorage * fs = cvOpenFileStorage(filename,0,CV_STORAGE_READ);
  if (!fs){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  CvFileNode * root = cvGetRootFileNode(fs);
  root = cvGetFileNodeByName(fs,root,"frames");
  CV_ASSERT(CV_NODE_IS_SEQ(root->tag));
  CvSeq * seq = root->data.seq; int total = seq->total;
  CvSeqReader reader; cvStartReadSeq( seq, &reader, 0 );
  data->rows=total;
  CvMat * image = cvCreateMat(240,240,CV_32F);
  for (int ii=0;ii<total;ii++){
    CvFileNode * node = (CvFileNode*)reader.ptr;
    if (!node){break;}
    CvSeq * seq2 = node->data.seq;
    CvSeqReader reader2; cvStartReadSeq( seq2, &reader2, 0 );
    const char * imgname = cvReadString((CvFileNode*)reader2.ptr,"");
    IplImage * img = cvLoadImage(imgname,0);
    CV_ASSERT(320*240==img->height*img->width);
    CvMat img_submat; cvGetSubRect(img,&img_submat,cvRect(0,0,240,240));
    CV_ASSERT(imsize==img_submat.height*img_submat.width);
    cvConvert(&img_submat,image);
    memcpy(data->data.fl+imsize*ii,image->data.ptr,imsize*sizeof(float));
    CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
  }
  cvReleaseMat(&image);
  cvReleaseFileStorage(&fs);
  __END__;
  return data;
}

CvMat * icvReadPrimateLabels(char * filename, const int max_samples)
{
  CV_FUNCNAME("icvReadPrimateLabels");
  CvMat * data = cvCreateMat(max_samples,6,CV_32F); cvZero(data);
  __BEGIN__;
  CvFileStorage * fs = cvOpenFileStorage(filename,0,CV_STORAGE_READ);
  if (!fs){fprintf(stderr,"file loading error: %s\n",filename);return 0;}
  CvFileNode * root = cvGetRootFileNode(fs);
  root = cvGetFileNodeByName(fs,root,"frames");
  CV_ASSERT(CV_NODE_IS_SEQ(root->tag));
  CvSeq * seq = root->data.seq; int total = seq->total;
  CvSeqReader reader; cvStartReadSeq( seq, &reader, 0 );
  data->rows=total;
  for (int ii=0;ii<total;ii++){
    CvFileNode * node = (CvFileNode*)reader.ptr;
    if (!node){break;}
    CvSeq * seq2 = node->data.seq;
    CvSeqReader reader2; cvStartReadSeq( seq2, &reader2, 0 );
    const char * imgname = cvReadString((CvFileNode*)reader2.ptr,"");
    CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int x1 = cvReadInt((CvFileNode*)reader2.ptr,0); CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int y1 = cvReadInt((CvFileNode*)reader2.ptr,0); CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int x2 = cvReadInt((CvFileNode*)reader2.ptr,0); CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int y2 = cvReadInt((CvFileNode*)reader2.ptr,0); CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int x3 = cvReadInt((CvFileNode*)reader2.ptr,0); CV_NEXT_SEQ_ELEM( seq2->elem_size, reader2 );
    int y3 = cvReadInt((CvFileNode*)reader2.ptr,0);
    // fprintf(stderr,"%s: (%d,%d) (%d,%d) (%d,%d)\n", imgname, x1, y1, x2, y2, x3, y3);
    CV_MAT_ELEM(*data,float,ii,0)=x1/240.f;
    CV_MAT_ELEM(*data,float,ii,1)=y1/240.f;
    CV_MAT_ELEM(*data,float,ii,2)=x2/240.f;
    CV_MAT_ELEM(*data,float,ii,3)=y2/240.f;
    CV_MAT_ELEM(*data,float,ii,4)=x3/240.f;
    CV_MAT_ELEM(*data,float,ii,5)=y3/240.f;
    CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
  }
  cvReleaseFileStorage(&fs);
  __END__;
  return data;
}
