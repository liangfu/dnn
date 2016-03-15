#include "network.h"

#include <stdio.h>

using namespace std;

CvNetwork::CvNetwork():m_solver(0),m_cnn(0)
{
  // m_clipHeight = height;
  // m_clipWidth  = width;
  // m_nNode = node;
  // m_connectNode = cNode;
  // m_learningRate = alpha;
  // m_max_iter = maxiter;
  // m_batch_size = batch_size;
}

CvNetwork::~CvNetwork(void)
{
  if( m_cnn ) { m_cnn->release( (CvCNNStatModel**)&m_cnn ); }
}

void CvNetwork::loadModel(string inFile)
{
  CvCNNLayer * layer = 0;
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  CvFileNode * root = cvGetRootFileNode( fs );
  CvFileNode * node = 0;
  char nodename[10]={0,};
  float lr_init = m_solver->lr_init();
  int decay_type = m_solver->decay_type();

  node = cvGetFileNodeByName(fs,root,"data");
  int n_input_planes = cvReadIntByName(fs,node,"n_input_planes");
  int input_height   = cvReadIntByName(fs,node,"input_height");
  int input_width    = cvReadIntByName(fs,node,"input_width");
  int n_output_planes = 0;
  int output_height   = 0;
  int output_width    = 0;
  
  m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
    CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel));

  root = cvGetFileNodeByName(fs,root,"layers");
  for (int ii=1;;ii++){
    sprintf(nodename,"layer-%d",ii);
    node = cvGetFileNodeByName(fs,root,nodename);
    if (!node){break;}
    const char * type = cvReadStringByName(fs,node,"type");
    const char * name = cvReadStringByName(fs,node,"name");
    if (!strcmp(type,"Convolution")){
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNConvolutionLayer(
        n_input_planes, input_height, input_width, n_output_planes, ksize,
        lr_init, decay_type, NULL, NULL );
      n_input_planes = n_output_planes;
      input_height = input_height-ksize+1;
      input_width = input_width-ksize+1;
    }else if (!strcmp(type,"SubSampling")){
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNSubSamplingLayer(
        n_input_planes, input_height, input_width, ksize,1,1,
        lr_init, decay_type, NULL);
      n_input_planes = n_output_planes;
      input_height = input_height/ksize;
      input_width = input_width/ksize;
    }else if (!strcmp(type,"FullConnect")){
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      const char * activation_desc = cvReadStringByName(fs,node,"activation_type");
      int activation_type = CV_CNN_RELU;
      n_input_planes = n_input_planes * input_height * input_width;
      if (!strcmp(activation_desc,"relu")){activation_type = CV_CNN_RELU;}else
      if (!strcmp(activation_desc,"tanh")){activation_type = CV_CNN_HYPERBOLIC;}else
      if (!strcmp(activation_desc,"logit")){activation_type = CV_CNN_LOGISTIC;}
      layer = cvCreateCNNFullConnectLayer(
        n_input_planes, n_output_planes, 1, 1, 
        lr_init, decay_type, activation_type, NULL );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }
    if (ii==1){m_cnn->network = cvCreateCNNetwork(layer); // add layer to network
    }else{m_cnn->network->add_layer( m_cnn->network, layer );}
  }

  if (fs){cvReleaseFileStorage(&fs);fs=0;}
}

void CvNetwork::loadWeights(string inFile)
{
  CvCNNConvolutionLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  CvFileNode * fnode = cvGetRootFileNode( fs );
  
  if (m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->layers;
  layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv1");
  layer->next_layer->next_layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv2");
  layer->next_layer->next_layer->next_layer->next_layer->weights = 
    (CvMat*)cvReadByName(fs,fnode,"softmax1");
  layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights = 
    (CvMat*)cvReadByName(fs,fnode,"softmax2");

  cvReleaseFileStorage(&fs);
}

void CvNetwork::saveWeights(string outFile)
{
  CvCNNConvolutionLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(outFile.c_str(),0,CV_STORAGE_WRITE);
  
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->layers;
  cvWrite(fs,"conv1",layer->weights);
  cvWrite(fs,"conv2",layer->next_layer->next_layer->weights);
  cvWrite(fs,"softmax1",layer->next_layer->next_layer->next_layer->next_layer->weights);
  cvWrite(fs,"softmax2",layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights);

  cvReleaseFileStorage(&fs);
}

void CvNetwork::train(CvMat *trainingData, CvMat *responseMat)
{
  int i, j;	
  CvCNNStatModelParams params;
  assert(CV_MAT_TYPE(trainingData->type)==CV_32F);

  params.cls_labels = cvCreateMat( 10, 10, CV_32FC1 );
  cvSet(params.cls_labels,cvScalar(1));
  params.etalons = cvCreateMat( 10, 10, CV_32FC1 );
  for(i=0;i<params.etalons->rows;i++){
  for(j=0;j<params.etalons->cols;j++){
    cvmSet(params.etalons,i,j,(double)-1.0);
  } cvmSet(params.etalons,i,i,(double)1.0);
  }

  params.network = m_cnn->network;
  params.start_iter=0;
  params.max_iter=m_solver->maxiter();
  params.batch_size = m_solver->batch_size();
  params.grad_estim_type=CV_CNN_GRAD_ESTIM_RANDOM;

  if (CV_MAT_TYPE(responseMat->type)!=CV_32S){
    CvMat * tmp = cvCreateMat(responseMat->rows,responseMat->cols,CV_32S);
    cvConvert(responseMat,tmp);
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,tmp,&params,0,0,0,0);
    cvReleaseMat(&tmp);
  }else{
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,responseMat,&params,0,0,0,0);
  }
}

float CvNetwork::evaluate(CvMat * testing, CvMat * expected, int nsamples)
{
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
  m_cnn->predict(m_cnn,samples,result);
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
}



