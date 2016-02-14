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
	cnn_train = 0;
}

ConvNN::ConvNN(int height, int width, int node, int cNode,
               double alpha, int maxiter)
{
	m_clipHeight = height;
	m_clipWidth  = width;
	m_nNode = node;
	m_connectNode = cNode;
	m_learningRate = alpha;
	m_max_iter = maxiter;
}

ConvNN::~ConvNN(void)
{

	if( cvGetErrStatus() < 0 )
	{
		if( m_cnn ) m_cnn->release( (CvCNNStatModel**)&m_cnn ); 
		if( cnn_train ) cnn_train->release( (CvCNNStatModel**)&cnn_train );
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

	CvCNNLayer     * layer;

	n_input_planes  = 1;
	input_height    = m_clipHeight;//m_clipHeight;
	input_width     = m_clipWidth;//m_clipWidth;
	n_output_planes = 6;
	//n_output_planes = 7;
	output_height   = (m_clipHeight-3)/2;
	output_width    = (m_clipWidth-3)/2;
	init_learn_rate = m_learningRate;//m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;//CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;//CV_CNN_DELTA_W_INCREASE_FIRSTORDER;
	//	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_FIRSTORDER;//CV_CNN_DELTA_W_INCREASE_LM;
	nsamples = 1;//NSAMPLES;
	maxiters = m_max_iter;//MAX_ITER;
	K = 5;
	a = 1;
	s = 1;

	CV_FUNCNAME("CNNTrainThread_Simard");
	__CV_BEGIN__;

	CV_CALL(m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
      CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));//,
		// NULL, NULL, NULL ));

	CV_CALL(layer = cvCreateCNNConvolutionLayer(
		n_input_planes, input_height, input_width, n_output_planes, K,a,s,
		init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, connect_mask, NULL ));
	CV_CALL(m_cnn->network = cvCreateCNNetwork( layer ));

	//n_input_planes  = 7;
	n_input_planes  = 6;
	input_height    = (m_clipHeight-3)/2;
	input_width     = (m_clipWidth-3)/2;
	n_output_planes = 50;
	//n_output_planes = 60;
	output_height   = (input_height-3)/2;
	output_width    = (input_width -3)/2;
	init_learn_rate = m_learningRate;
	K = 5;


	CV_CALL(layer = cvCreateCNNConvolutionLayer(
		n_input_planes, input_height, input_width, n_output_planes, K,a,s,
		init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, connect_mask, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = n_output_planes * output_height* output_width;
	n_output_planes = m_connectNode;
	init_learn_rate = m_learningRate;
	a = 1;
	s = 1;
	CV_CALL(layer = cvCreateCNNFullConnectLayer( n_input_planes, n_output_planes,   \
		a, s, init_learn_rate, learn_type, delta_w_increase_type,nsamples, maxiters, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = m_connectNode;
	n_output_planes = 12;
	init_learn_rate = m_learningRate;
	a = 1;
	s = 1;
	CV_CALL(layer = cvCreateCNNFullConnectLayer( n_input_planes, n_output_planes,   \
		a, s, init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, NULL ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));
	__CV_END__;

}

void ConvNN::writeCNNParams(string outFile)
{
	FILE *fpweights;
	CvCNNConvolutionLayer* templayer;
	int i,j;

	if(m_cnn == NULL)
	{
		printf("error! CNN has not been built yet\n");
		exit(0);
	}

	templayer=(CvCNNConvolutionLayer*)m_cnn->network->layers;
	if((fpweights=fopen(outFile.c_str(),"w"))==NULL)
	{
		printf("Can not open file to store CNN weights\n");
		exit(0);
	}
	//str.Format("weight size: %d,%d",templayer->weights->rows,templayer->weights->cols);
	fprintf(fpweights,"weights of 1st layer weight size: %d,%d\n",templayer->weights->rows,templayer->weights->cols);
	for(i=0;i<templayer->weights->rows;i++)
	{
		for(j=0;j<templayer->weights->cols;j++)
		{
			fprintf(fpweights,"%f\n",cvmGet(templayer->weights,i,j));
		}
	}
	templayer=(CvCNNConvolutionLayer*)m_cnn->network->layers->next_layer;
	fprintf(fpweights,"weights of 2nd layer weight size: %d,%d\n",templayer->weights->rows,templayer->weights->cols);
	//fprintf(fpweights,"weights of 2nd layer %s:\n",str);
	for(i=0;i<templayer->weights->rows;i++)
	{
		for(j=0;j<templayer->weights->cols;j++)
		{
			fprintf(fpweights,"%f\n",cvmGet(templayer->weights,i,j));
		}
	}
	templayer=(CvCNNConvolutionLayer*)m_cnn->network->layers->next_layer->next_layer;
	fprintf(fpweights,"weights of 3rd layer weight size: %d,%d\n",templayer->weights->rows,templayer->weights->cols);
	for(i=0;i<templayer->weights->rows;i++)
	{
		for(j=0;j<templayer->weights->cols;j++)
		{
			fprintf(fpweights,"%f\n",cvmGet(templayer->weights,i,j));
		}
	}
	templayer=(CvCNNConvolutionLayer*)m_cnn->network->layers->next_layer->next_layer->next_layer;
	fprintf(fpweights,"weights of 4th layer weight size: %d,%d\n",templayer->weights->rows,templayer->weights->cols);
	//fprintf(fpweights,"weights of 2nd layer %s:\n",str);
	for(i=0;i<templayer->weights->rows;i++)
	{
		for(j=0;j<templayer->weights->cols;j++)
		{
			fprintf(fpweights,"%f\n",cvmGet(templayer->weights,i,j));
		}
	}

	fclose(fpweights);
}

void ConvNN::LoadCNNParams(string inFile/*string inFile, int nSample, float maxIter, 
									  int input_planes, int input_height, int input_width, int output_planes, int output_height,
									  int output_width, float learn_rate, int learn_type, int delat_w_increase_type, 
									  int  P_K, int P_a, int P_s*/)
{
	float a,s;
	int n_input_planes, input_height, input_width;
	int n_output_planes, output_height, output_width;
	int learn_type;
	int delta_w_increase_type;
	double init_learn_rate;
	int nsamples;
	int maxiters;
	int K;

	CvMat* weights_C1;
	CvMat* weights_C2;
	CvMat* weights_F1;
	CvMat* weights_F2;

	CvCNNLayer* layer = 0;
	CvMat* connect_mask = 0;

	weights_C1 = cvCreateMat( 6*26, 1, CV_32FC1 );
	weights_C2 = cvCreateMat( 50*(6*25+1), 1, CV_32FC1 );

	input_height = ((m_clipHeight-3)/2-3)/2;
	input_width  = ((m_clipWidth -3)/2-3)/2;

	weights_F1 = cvCreateMat( m_connectNode, 50*input_height*input_width+1, CV_32FC1 );
	weights_F2 = cvCreateMat( m_nNode, m_connectNode+1, CV_32FC1 );

	FILE *fpweights;
	int i, j, ti, tj;


	if((fpweights=fopen(inFile.c_str(),"r")) ==NULL)
	{
		printf("Can not open the weight file\n");
		exit(0);
	}

	fscanf(fpweights,"weights of 1st layer weight size: %d,%d:\n",&ti,&tj);
	for(i=0;i<156;i++)
	{
		for(j=0;j<1;j++)
		{
			fscanf(fpweights,"%f\n",&weights_C1->data.fl[i*1+j]);
		}
	}
	fscanf(fpweights,"weights of 2nd layer weight size: %d,%d:\n",&ti,&tj);

	//for(i=0;i<10560;i++)
	for(i=0;i<7550;i++)
	{
		for(j=0;j<1;j++)
		{
			fscanf(fpweights,"%f\n",&weights_C2->data.fl[i*1+j]);
		}
	}
	fscanf(fpweights,"weights of 3rd layer weight size: %d,%d:\n",&ti,&tj);
	//int rowLen = (60*input_height*input_width+1);
	int rowLen = (50*input_height*input_width+1);

	for(i=0;i<m_connectNode;i++)
	{
		for(j=0;j<(50*input_height*input_width+1);j++)
			//for(j=0;j<(60*input_height*input_width+1);j++)
		{
			fscanf(fpweights,"%f\n",&weights_F1->data.fl[i*rowLen+j]);
		}
	}
	fscanf(fpweights,"weights of 4th layer weight size: %d,%d:\n",&ti,&tj);
	for(i=0;i<m_nNode;i++)
	{
		for(j=0;j<(m_connectNode+1);j++)
		{
			fscanf(fpweights,"%f\n",&weights_F2->data.fl[i*(m_connectNode+1)+j]);
		}
	}
	fclose(fpweights);

	n_input_planes  = 1;
	input_height    = m_clipHeight;
	input_width     = m_clipWidth;
	n_output_planes = 6;
	//n_output_planes = 7;
	output_height   = (m_clipHeight-3)/2;
	output_width    = (m_clipWidth-3)/2;
	init_learn_rate = m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
	nsamples = 10;//m_nsamples;
	maxiters = m_max_iter;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;
	K = 5;
	a = 1.0f;
	s = 1.0f;

	CV_FUNCNAME("OnButtonCNNTest");
	__CV_BEGIN__;

	CV_CALL(m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
      CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel)));// ,
		// NULL, NULL, NULL ));

	CV_CALL(layer = cvCreateCNNConvolutionLayer(
		n_input_planes, input_height, input_width, n_output_planes, K,a,s,
		init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, connect_mask, weights_C1 ));
	CV_CALL(m_cnn->network = cvCreateCNNetwork( layer ));

	//n_input_planes  = 7;
	n_input_planes  = 6;
	input_height    = output_height;
	input_width     = output_width;
	n_output_planes = 50;
	//n_output_planes = 60;
	output_height   = (input_height-3)/2;
	output_width    = (input_width-3)/2;
	init_learn_rate = m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;
	K = 5;
	CV_CALL(layer = cvCreateCNNConvolutionLayer(
		n_input_planes, input_height, input_width, n_output_planes, K,a,s,
		init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, connect_mask, weights_C2 ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = n_output_planes*output_height*output_width;
	n_output_planes = m_connectNode;
	init_learn_rate = m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;
	a = 1;
	s = 1;
	CV_CALL(layer = cvCreateCNNFullConnectLayer( n_input_planes, n_output_planes,   \
		a, s, (float)init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, weights_F1 ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	n_input_planes  = m_connectNode;
	n_output_planes =  m_nNode ;
	init_learn_rate = m_learningRate;
	learn_type = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
	delta_w_increase_type = CV_CNN_DELTA_W_INCREASE_LM;
	a = 1;
	s = 1;
	CV_CALL(layer = cvCreateCNNFullConnectLayer( n_input_planes, n_output_planes,   \
		a, s, (float)init_learn_rate, learn_type, delta_w_increase_type, nsamples, maxiters, weights_F2 ));
	CV_CALL(m_cnn->network->add_layer( m_cnn->network, layer ));

	__CV_END__;


	cvReleaseMat( &weights_C1 );
	cvReleaseMat( &weights_C2 );
	cvReleaseMat( &weights_F1 );
	cvReleaseMat( &weights_F2 );

	cvReleaseMat( &connect_mask );
}

void ConvNN::trainNN(CvMat *trainingData, CvMat *responseMat,
                     CvMat *testingData, CvMat *expectedMat)
{
	int i, j;	

	CvCNNStatModelParams params;
	params.cls_labels = cvCreateMat( 12, 12, CV_32FC1 );

	params.etalons = cvCreateMat( 12, 12, CV_32FC1 );
	for(i=0;i<12;i++){
		for(j=0;j<12;j++){
			cvmSet(params.etalons,i,j,(double)-1.0);
    }
		cvmSet(params.etalons,i,i,(double)1.0);
	}

	for(i=0;i<12;i++){
		for(j=0;j<12;j++){
			cvmSet(params.cls_labels,i,j,(double)1.0);
    }
	}
	params.network = m_cnn->network;
	params.start_iter=0;
	params.max_iter=m_max_iter;
	params.grad_estim_type=CV_CNN_GRAD_ESTIM_RANDOM;//CV_CNN_GRAD_ESTIM_BY_WORST_IMG;

	m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,responseMat,&params,0,0,0,0);
}

void ConvNN::predictNN(CvMat *trainingData, CvMat **responseMat)
{
  //icvCNNModelPredict( m_cnn,trainingData, responseMat);
}
