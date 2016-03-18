#include "dram.h"
// #include "global.h"
// #include "ml.h"

#include <stdio.h>

using namespace std;

DRAM::DRAM(void)
{
  m_cnn = 0;
  // cnn_train = 0;
  m_batch_size = 1;
}

DRAM::DRAM(int height, int width, int cNode,int node, 
               double alpha, int maxiter, int batch_size)
{
  m_clipHeight = height;
  m_clipWidth  = width;
  m_nNode = node;
  m_connectNode = cNode;
  m_learningRate = alpha;
  m_max_iter = maxiter;
  m_batch_size = batch_size;
}

DRAM::~DRAM(void)
{
  if( cvGetErrStatus() < 0 ) {
    if( m_cnn ) { m_cnn->release( (CvCNNStatModel**)&m_cnn ); }
    // if( cnn_train ) cnn_train->release( (CvCNNStatModel**)&cnn_train );
  }
}

void DRAM::createNetwork()
{
  CvMat* connect_mask = 0;
  int n_input_planes, input_height, input_width;
  int n_output_planes, output_height, output_width;
  int learn_type;
  int delta_w_increase_type;
  float init_learn_rate;
  int maxiters;
  int K,N,S;
  int sub_samp_size;
  int activation_type;
  CvCNNLayer     * layer;
  int n_hiddens; int seq_length;

  learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
  delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_FIRSTORDER;
  maxiters = m_max_iter;
  init_learn_rate = m_learningRate;
  K = 5;
  N = 1; // number of glimpse
  S = 1; // number of targets

  CV_FUNCNAME("createNetwork");
  __CV_BEGIN__;

  CV_CALL(m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
    CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));

  //-------------------------------------------------------
  // Context Network
  //-------------------------------------------------------
  n_input_planes  = 1;
  input_height    = m_clipHeight;
  input_width     = m_clipWidth;
  n_output_planes = 6;
  output_height   = input_height-K+1;
  output_width    = input_width-K+1;
  sub_samp_size=2;
  CV_CALL(layer = cvCreateCNNConvolutionLayer( "conv1", 
    n_input_planes, input_height, input_width, n_output_planes, K,
    init_learn_rate, learn_type, connect_mask, NULL ));
  CV_CALL(m_cnn->network = cvCreateCNNetwork( layer ));
  CV_CALL(layer = cvCreateCNNSubSamplingLayer( "pool1", 
      n_output_planes, output_height, output_width, sub_samp_size,1,1,
      init_learn_rate, learn_type, NULL));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  n_input_planes  = n_output_planes;
  input_height    = output_height/sub_samp_size;
  input_width     = output_width/sub_samp_size;
  n_output_planes = 16;
  output_height   = input_height-K+1;
  output_width    = input_width-K+1; 
  CV_CALL(layer = cvCreateCNNConvolutionLayer( "conv2", 
    n_input_planes, input_height, input_width, n_output_planes, K,
    init_learn_rate, learn_type, connect_mask, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  CV_CALL(layer = cvCreateCNNSubSamplingLayer( "pool2", 
      n_output_planes, output_height, output_width, sub_samp_size,1,1,
      init_learn_rate, learn_type, NULL));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  output_height   = output_height/sub_samp_size;
  output_width    = output_width/sub_samp_size; 

  // We assign a fixed number of glimpses, N, for each target. Assuming S targets in
  // an image, the model would be trained with N × (S + 1) glimpses
  // for (int nn = 0; nn < N; nn++){
  // for (int ss = 0; ss < S; ss++){
  
  //-------------------------------------------------------
  // Recurrent Network II
  //-------------------------------------------------------
  n_input_planes  = n_output_planes * output_height* output_width;
  n_output_planes = 128;
  activation_type = CV_CNN_HYPERBOLIC;
  n_hiddens = cvRound(exp2((log2(n_input_planes)+log2(n_output_planes))*.5f));
  seq_length = N*(S+1);
  CV_CALL(layer = cvCreateCNNRecurrentLayer( "rnn2", 0, 
      n_input_planes, n_output_planes, n_hiddens, seq_length, 0, 
      init_learn_rate, learn_type, activation_type, NULL, NULL, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  //-------------------------------------------------------
  // Emission Network
  //-------------------------------------------------------
  n_input_planes  = n_output_planes;
  n_output_planes = 2; // location
  activation_type = CV_CNN_LOGISTIC;
  CV_CALL(layer = cvCreateCNNFullConnectLayer( "fc1", 
      n_input_planes, n_output_planes, 1, 1, 
      init_learn_rate, learn_type, activation_type, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  //-------------------------------------------------------
  // Glimpse Network
  //-------------------------------------------------------
  n_input_planes  = 1;
  output_height    = 48;//m_clipHeight;
  output_width     = 48;//m_clipWidth;
  CV_CALL(layer = cvCreateCNNImgCroppingLayer( "crop1", 
    n_input_planes, output_height, output_width, m_cnn->network->first_layer,
    init_learn_rate, learn_type));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  
  n_input_planes  = 1;
  input_height    = output_height;
  input_width     = output_width;
  n_output_planes = 6;
  output_height   = input_height-K+1;
  output_width    = input_width-K+1;
  sub_samp_size=2;
  CV_CALL(layer = cvCreateCNNConvolutionLayer( "conv3",
    n_input_planes, input_height, input_width, n_output_planes, K,
    init_learn_rate, learn_type, connect_mask, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  CV_CALL(layer = cvCreateCNNSubSamplingLayer( "pool3", 
      n_output_planes, output_height, output_width, sub_samp_size,1,1,
      init_learn_rate, learn_type, NULL));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  n_input_planes  = n_output_planes;
  input_height    = output_height/sub_samp_size;
  input_width     = output_width/sub_samp_size;
  n_output_planes = 16;
  output_height   = input_height-K+1;
  output_width    = input_width-K+1; 
  CV_CALL(layer = cvCreateCNNConvolutionLayer( "conv4", 
    n_input_planes, input_height, input_width, n_output_planes, K,
    init_learn_rate, learn_type, connect_mask, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  CV_CALL(layer = cvCreateCNNSubSamplingLayer( "pool4", 
      n_output_planes, output_height, output_width, sub_samp_size,1,1,
      init_learn_rate, learn_type, NULL));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
  output_height   = output_height/sub_samp_size;
  output_width    = output_width/sub_samp_size; 
  
  //-------------------------------------------------------
  // Recurrent Network I
  //-------------------------------------------------------
  n_input_planes  = n_output_planes * output_height* output_width;
  n_output_planes = 128;
  activation_type = CV_CNN_HYPERBOLIC; // CV_CNN_RELU;
  CV_CALL(layer = cvCreateCNNRecurrentLayer( "rnn1", 0, 
      n_input_planes, n_output_planes, n_hiddens, seq_length, 0, 
      init_learn_rate, learn_type, activation_type, NULL, NULL, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  n_input_planes  = n_output_planes;
  n_output_planes = 10;
  activation_type = CV_CNN_LOGISTIC;
  CV_CALL(layer = cvCreateCNNFullConnectLayer( "rnn1", 
      n_input_planes, n_output_planes, 1, 1, 
      init_learn_rate, learn_type, activation_type, NULL ));
  CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  // } // ss
  // } // nn

  __CV_END__;
}

void DRAM::writeNetworkParams(string outFile)
{
  CvCNNConvolutionLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(outFile.c_str(),0,CV_STORAGE_WRITE);
  
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->first_layer;
  cvWrite(fs,"conv1",layer->weights);
  cvWrite(fs,"conv2",layer->next_layer->next_layer->weights);
  cvWrite(fs,"softmax1",layer->next_layer->next_layer->next_layer->next_layer->weights);
  cvWrite(fs,"softmax2",layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights);

  cvReleaseFileStorage(&fs);
}

void DRAM::readNetworkParams(string inFile)
{
  CvCNNConvolutionLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  CvFileNode * fnode = cvGetRootFileNode( fs );
  
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->first_layer;
  layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv1");
  layer->next_layer->next_layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv2");
  layer->next_layer->next_layer->next_layer->next_layer->weights = 
    (CvMat*)cvReadByName(fs,fnode,"softmax1");
  layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights = 
    (CvMat*)cvReadByName(fs,fnode,"softmax2");

  cvReleaseFileStorage(&fs);
}

void DRAM::trainNetwork(CvMat *trainingData, CvMat *responseMat)
{
  int i, j;	
  CvCNNStatModelParams params;
  assert(CV_MAT_TYPE(trainingData->type)==CV_32F);

  params.cls_labels = cvCreateMat( 10, 10, CV_32FC1 );
  cvSet(params.cls_labels,cvScalar(1));
  params.etalons = cvCreateMat( 10, m_nNode, CV_32FC1 );
  for(i=0;i<params.etalons->rows;i++){
  for(j=0;j<params.etalons->cols;j++){
    cvmSet(params.etalons,i,j,(double)-1.0);
  } cvmSet(params.etalons,i,i,(double)1.0);
  }

  params.network = m_cnn->network;
  params.start_iter=0;
  params.max_iter=m_max_iter;
  params.batch_size = m_batch_size;
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

void DRAM::predictNN(CvMat *trainingData, CvMat **responseMat)
{
  //icvCNNModelPredict( m_cnn,trainingData, responseMat);
}


