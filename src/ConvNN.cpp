#include "ConvNN.h"
// #include "global.h"
// #include "ml.h"

#include <stdio.h>

using namespace std;

CNNIO::CNNIO(void)
{
	output = NULL;
	fpredict =NULL;
}


CNNIO::~CNNIO(void)
{
	if(output != NULL)
	{
		for(int k=0; k< 5; k++)
			cvReleaseMat(&output[k]);

		cvFree(output);
	}
}

void CNNIO::init(int outNode, int width, int height, ConvNN *CNN)
{
	output = (CvMat**)cvAlloc( (outNode+1)*sizeof(CvMat*) );
	memset( output, 0, (outNode+1)*sizeof(CvMat*) );
	output[0] = cvCreateMat( height*width,1,CV_32FC1 );
	CvCNNLayer * layer;

	int k,i;
	int n_layers = CNN->m_cnn->network->n_layers;

	for( k = 0, layer = CNN->m_cnn->network->layers; k < n_layers; k++, layer = layer->next_layer )
	{
		output[k+1] = cvCreateMat( layer->n_output_planes*layer->output_height*
			layer->output_width, 1, CV_32FC1 );
	}

	CNN->m_cnn->cls_labels = cvCreateMat(1,NNODE,CV_32FC1);
	for(i=0;i<NNODE;i++)
		CNN->m_cnn->cls_labels->data.i[i]=i;
}



ConvNN::ConvNN(void)
{
	m_cnn = 0;
	// cnn_train = 0;
    m_batch_size = 1;
}

ConvNN::ConvNN(int height, int width, int cNode,int node, 
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

ConvNN::~ConvNN(void)
{

	if( cvGetErrStatus() < 0 )
	{
		if( m_cnn ) m_cnn->release( (CvCNNStatModel**)&m_cnn ); 
		// if( cnn_train ) cnn_train->release( (CvCNNStatModel**)&cnn_train );
	}

	

}

void ConvNN::createCNN()/*(int nSample, float maxIter, 
						int input_planes, int input_height, int input_width, int output_planes, int output_height,
						int output_width, float learn_rate, int learn_type, int delat_w_increase_type, 
						int  P_K, int P_a, int P_s*/
{
	CvMat* connect_mask = 0;
	float a,s;
	int n_input_planes, input_height, input_width;
	int n_output_planes, output_height, output_width;
	int learn_type;
	int delta_w_increase_type;
	float init_learn_rate;
	int nsamples;
	int maxiters;
	int K;
  int sub_samp_size;
  int activation_type;
  
	CvCNNLayer     * layer;

	n_input_planes  = 1;
	input_height    = m_clipHeight;
	input_width     = m_clipWidth;
	n_output_planes = 6;
	K = 5;
	output_height   = input_height-K+1; // (m_clipHeight-3)/2;//
	output_width    = input_width-K+1; // (m_clipWidth-3)/2;//
	init_learn_rate = m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;//CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY;
	// delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;//CV_CNN_DELTA_W_INCREASE_FIRSTORDER;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_FIRSTORDER;//CV_CNN_DELTA_W_INCREASE_LM;
	nsamples = 1;//NSAMPLES;
	maxiters = m_max_iter;//MAX_ITER;
	a = 1;
	s = 1;

	CV_FUNCNAME("CNNTrainThread_Simard");
	__CV_BEGIN__;

	CV_CALL(m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
      CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));//,
		// NULL, NULL, NULL ));

  // 20 @ 28x28
	CV_CALL(layer = cvCreateCNNConvolutionLayer(
    n_input_planes, input_height, input_width, n_output_planes, K,
		init_learn_rate, learn_type,
    connect_mask, NULL ));
	CV_CALL(m_cnn->network = cvCreateCNNetwork( layer ));

  // 20 @ 14x14
  sub_samp_size=2;
	CV_CALL(layer = cvCreateCNNSubSamplingLayer(
      n_output_planes, output_height, output_width, sub_samp_size,a,s,
      init_learn_rate, learn_type, NULL));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = n_output_planes;
	input_height    = output_height/sub_samp_size;//(m_clipHeight-3)/2;//
	input_width     = output_width/sub_samp_size;//(m_clipWidth-3)/2;//
	n_output_planes = 16;
	K = 5;
	output_height   = input_height-K+1; // (input_height-3)/2;//
	output_width    = input_width-K+1; // (input_width -3)/2;//
	init_learn_rate = m_learningRate;

  // 50 @ 14x14
	CV_CALL(layer = cvCreateCNNConvolutionLayer(
    n_input_planes, input_height, input_width, n_output_planes, K,
		init_learn_rate, learn_type,
    connect_mask, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

  sub_samp_size=2;
	CV_CALL(layer = cvCreateCNNSubSamplingLayer(
      n_output_planes, output_height, output_width, sub_samp_size,a,s,
      init_learn_rate, learn_type, NULL));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	output_height   = output_height/sub_samp_size; // (input_height-3)/2;
	output_width    = output_width/sub_samp_size; // (input_width -3)/2;
  
	n_input_planes  = n_output_planes * output_height* output_width;
	n_output_planes = m_connectNode;
	init_learn_rate = m_learningRate;
	a = 1;
	s = 1;
  activation_type = CV_CNN_HYPERBOLIC; // CV_CNN_RELU;
	CV_CALL(layer = cvCreateCNNFullConnectLayer(
      n_input_planes, n_output_planes, a, s, 
      init_learn_rate, learn_type, activation_type, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = m_connectNode;
	n_output_planes = m_nNode;
	init_learn_rate = m_learningRate;
	a = 1;
	s = 1;
  activation_type = CV_CNN_HYPERBOLIC;
	CV_CALL(layer = cvCreateCNNFullConnectLayer(
      n_input_planes, n_output_planes, a, s, 
      init_learn_rate, learn_type, activation_type, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	__CV_END__;

}

void ConvNN::writeCNNParams(string outFile)
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

void ConvNN::readCNNParams(string inFile)
{
  CvCNNConvolutionLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  CvFileNode * fnode = cvGetRootFileNode( fs );
  
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->layers;
  layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv1");
  layer->next_layer->next_layer->weights = (CvMat*)cvReadByName(fs,fnode,"conv2");
  layer->next_layer->next_layer->next_layer->next_layer->weights = (CvMat*)cvReadByName(fs,fnode,"softmax1");
  layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights = (CvMat*)cvReadByName(fs,fnode,"softmax2");

  cvReleaseFileStorage(&fs);
}

void ConvNN::trainNN(CvMat *trainingData, CvMat *responseMat,
                     CvMat *testingData, CvMat *expectedMat)
{
  int i, j;	

  CvCNNStatModelParams params;
  params.cls_labels = cvCreateMat( 10, 10, CV_32FC1 );

  params.etalons = cvCreateMat( 10, m_nNode, CV_32FC1 );
  for(i=0;i<params.etalons->rows;i++){
  for(j=0;j<params.etalons->cols;j++){
    cvmSet(params.etalons,i,j,(double)-1.0);
    // cvmSet(params.etalons,i,j,(double)0);
  }
  cvmSet(params.etalons,i,i,(double)1.0);
  }

  cvSet(params.cls_labels,cvScalar(1));
  // for(i=0;i<params.cls_labels->rows;i++){
  // for(j=0;j<params.cls_labels->cols;j++){
  // 	cvmSet(params.cls_labels,i,j,(double)1.0);
  // }
  // }
  params.network = m_cnn->network;
  params.start_iter=0;
  params.max_iter=m_max_iter;
  params.batch_size = m_batch_size;
  params.grad_estim_type=CV_CNN_GRAD_ESTIM_RANDOM;//CV_CNN_GRAD_ESTIM_BY_WORST_IMG;

  // CvMat * trainingData_scaled = cvCreateMat(trainingData->rows,trainingData->cols,CV_32F);
  // CvMat * testingData_scaled = cvCreateMat(testingData->rows,testingData->cols,CV_32F);
  // cvSubS(trainingData,cvScalar(-127.5f),trainingData_scaled);
  // cvScale(trainingData,trainingData_scaled,10./127.5f);
  // cvScale(testingData,testingData_scaled,1./255.f);

  if (CV_MAT_TYPE(responseMat->type)!=CV_32S){
    CvMat * tmp = cvCreateMat(responseMat->rows,responseMat->cols,CV_32S);
    cvConvert(responseMat,tmp);
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,tmp,&params,0,0,0,0);
    cvReleaseMat(&tmp);
  }else{
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,responseMat,&params,0,0,0,0);
  }

  // cvReleaseMat(&trainingData_scaled);
  // cvReleaseMat(&testingData_scaled);
}

void ConvNN::predictNN(CvMat *trainingData, CvMat **responseMat)
{
  //icvCNNModelPredict( m_cnn,trainingData, responseMat);
}
