/* [ mlcnn.cpp ] - Machine Learning - Convolutional Neural Network
 * ===========================================================================
 * Author: Xutao Lv <xutaolv@mail.mizzou.edu>
 * Revision: 1.0
 * Date: 05 October 2008 - (First: 29 August 2008)
 * ===========================================================================
 * Some of the functions are from OpenCV, so you should consent to its license.
 */

// #include "cv.h"
// #include "highgui.h"
// #include "cxcore.h"
// #include "_ml.h"
// #include "ml.h"
// #include "global.h"
#include "precomp.hpp"
#include <iostream>
#include <string>

using namespace std;

//sigmoid function
#define SIG(p) (1.7159*tanh(0.66666667*p))
#define DSIG(p) (0.66666667/1.7159*(1.7159+(p))*(1.7159-(p)))  // derivative of the sigmoid
/****************************************************************************************\
*                         Auxilary functions declarations                                *
 \****************************************************************************************/
/*---------------------- functions for the CNN classifier ------------------------------*/
static void icvCNNModelRelease(CvCNNStatModel** cnn_model);

static void icvTrainCNNetwork( CvCNNetwork* network,
                               const float** images,
                               const CvMat* responses,
                               const CvMat* etalons,
                               int grad_estim_type,
                               int max_iter,
                               int start_iter );

/*------------------------- functions for the CNN network ------------------------------*/
static void icvCNNetworkAddLayer( CvCNNetwork* network, CvCNNLayer* layer );
static void icvCNNetworkRelease( CvCNNetwork** network );

/* In all layer functions we denote input by X and output by Y, where
   X and Y are column-vectors, so that
   length(X)==<n_input_planes>*<input_height>*<input_width>,
   length(Y)==<n_output_planes>*<output_height>*<output_width>.
*/
/*------------------------ functions for convolutional layer ---------------------------*/
static void icvCNNConvolutionRelease( CvCNNLayer** p_layer );
static void icvCNNConvolutionForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNConvolutionBackward( CvCNNLayer*  layer, int t,
    const CvMat* X, CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2, CvMat* d2E_dX2 );
/*------------------------ functions for sub-sampling layer ----------------------------*/
static void icvCNNSubSamplingRelease( CvCNNLayer** p_layer );
static void icvCNNSubSamplingForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNSubSamplingBackward( CvCNNLayer*  layer, int t,
    const CvMat* X, CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2, CvMat* d2E_dX2 );
/*------------------------ functions for full connected layer --------------------------*/
static void icvCNNFullConnectRelease( CvCNNLayer** p_layer );
static void icvCNNFullConnectForward( CvCNNLayer* layer, const CvMat* X, CvMat* Y );
static void icvCNNFullConnectBackward( CvCNNLayer* layer, int,
    const CvMat*, CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2, CvMat* d2E_dX2 );
/****************************************************************************************\
*                             Functions implementations                                  *
 \****************************************************************************************/
#define ICV_CHECK_CNN_NETWORK(network)                                                  \
{                                                                                       \
    CvCNNLayer* first_layer, *layer, *last_layer;                                       \
    int n_layers, i;                                                                    \
    if( !network )                                                                      \
        CV_ERROR( CV_StsNullPtr,                                                        \
        "Null <network> pointer. Network must be created by user." );                   \
    n_layers = network->n_layers;                                                       \
    first_layer = last_layer = network->layers;                                         \
    for( i = 0, layer = first_layer; i < n_layers && layer; i++ )                       \
    {                                                                                   \
        if( !ICV_IS_CNN_LAYER(layer) )                                                  \
            CV_ERROR( CV_StsNullPtr, "Invalid network" );                               \
        last_layer = layer;                                                             \
        layer = layer->next_layer;                                                      \
    }                                                                                   \
                                                                                        \
    if( i == 0 || i != n_layers || first_layer->prev_layer || layer )                   \
        CV_ERROR( CV_StsNullPtr, "Invalid network" );                                   \
                                                                                        \
    if( first_layer->n_input_planes != 1 )                                              \
        CV_ERROR( CV_StsBadArg, "First layer must contain only one input plane" );      \
                                                                                        \
    /*if( img_size != first_layer->input_height*first_layer->input_width )                \
        CV_ERROR( CV_StsBadArg, "Invalid input sizes of the first layer" );*/             \
                                                                                        \
    if( params->etalons->cols != last_layer->n_output_planes*                           \
        last_layer->output_height*last_layer->output_width )                            \
        CV_ERROR( CV_StsBadArg, "Invalid output sizes of the last layer" );             \
}

#define ICV_CHECK_CNN_MODEL_PARAMS(params)                                              \
{                                                                                       \
    if( !params )                                                                       \
        CV_ERROR( CV_StsNullPtr, "Null <params> pointer" );                             \
                                                                                        \
    if( !ICV_IS_MAT_OF_TYPE(params->etalons, CV_32FC1) )                                \
        CV_ERROR( CV_StsBadArg, "<etalons> must be CV_32FC1 type" );                    \
    if( params->etalons->rows != params->cls_labels->cols )                          \
        CV_ERROR( CV_StsBadArg, "Invalid <etalons> size" );                             \
                                                                                        \
    if( params->grad_estim_type != CV_CNN_GRAD_ESTIM_RANDOM &&                          \
        params->grad_estim_type != CV_CNN_GRAD_ESTIM_BY_WORST_IMG )                     \
        CV_ERROR( CV_StsBadArg, "Invalid <grad_estim_type>" );                          \
                                                                                        \
    if( params->start_iter < 0 )                                                        \
        CV_ERROR( CV_StsBadArg, "Parameter <start_iter> must be positive or zero" );    \
                                                                                        \
    if( params->max_iter < 1 )                                                \
        params->max_iter = 1;                                                 \
}

/****************************************************************************************\
*                              Classifier functions                                      *
\****************************************************************************************/
// the cvCreateStatModel function can be found nowhere, so I try to implement it - lxts
ML_IMPL CvCNNStatModel*	
cvCreateCNNStatModel(
        int flag, int size,
        CvCNNStatModelRelease release,
		CvCNNStatModelPredict predict,
		CvCNNStatModelUpdate update)
{
	CvCNNStatModel *p_model;
	CV_FUNCNAME("cvCreateStatModel");
	__CV_BEGIN__;

	//add the implementation here
	CV_CALL(p_model = (CvCNNStatModel*) cvAlloc(sizeof(*p_model)));
	memset(p_model, 0, sizeof(*p_model));
	p_model->release = icvCNNModelRelease; //release;
	p_model->update = NULL;
	p_model->predict = NULL;

	__CV_END__;

	if (cvGetErrStatus() < 0) {
		CvCNNStatModel* base_ptr = (CvCNNStatModel*) p_model;

		if (p_model && p_model->release)
			p_model->release(&base_ptr);
		else
			cvFree(&p_model);
		p_model = 0;
	}

	return (CvCNNStatModel*) p_model;
	return NULL;
}

#if 0
const float**
cvGetTrainSamples( const CvMat* train_data, int tflag,
                   const CvMat* var_idx, const CvMat* sample_idx,
                   int* _var_count, int* _sample_count,
                   bool always_copy_data )
{
    float** samples = 0;
    
    CV_FUNCNAME( "cvGetTrainSamples" );

    __CV_BEGIN__;
    
    int i, j, var_count, sample_count, s_step, v_step;
    bool copy_data;
    const float* data;
    const int *s_idx, *v_idx;

    if( !CV_IS_MAT(train_data) )
        CV_ERROR( CV_StsBadArg, "Invalid or NULL training data matrix" );

    var_count = var_idx ? var_idx->cols + var_idx->rows - 1 :
                tflag == CV_ROW_SAMPLE ? train_data->cols : train_data->rows;
    sample_count = sample_idx ? sample_idx->cols + sample_idx->rows - 1 :
                   tflag == CV_ROW_SAMPLE ? train_data->rows : train_data->cols;

    if( _var_count )
        *_var_count = var_count;

    if( _sample_count )
        *_sample_count = sample_count;

    copy_data = tflag != CV_ROW_SAMPLE || var_idx || always_copy_data;
    
    CV_CALL( samples = (float**)cvAlloc(sample_count*sizeof(samples[0]) +
                (copy_data ? 1 : 0)*var_count*sample_count*sizeof(samples[0][0])) );
    data = train_data->data.fl;
    s_step = train_data->step / sizeof(samples[0][0]);
    v_step = 1;
    s_idx = sample_idx ? sample_idx->data.i : 0;
    v_idx = var_idx ? var_idx->data.i : 0;

	if (!copy_data) {
		for (i = 0; i < sample_count; i++)
			samples[i] = (float*) (data + (s_idx ? s_idx[i] : i) * s_step);
	} else {
		samples[0] = (float*) (samples + sample_count);
		if (tflag != CV_ROW_SAMPLE)
			CV_SWAP(s_step, v_step, i);

		for (i = 0; i < sample_count; i++) {
			float* dst = samples[i] = samples[0] + i * var_count;
			const float* src = data + (s_idx ? s_idx[i] : i) * s_step;

			if (!v_idx)
				for (j = 0; j < var_count; j++)
					dst[j] = src[j * v_step];
			else
				for (j = 0; j < var_count; j++)
					dst[j] = src[v_idx[j] * v_step];
		}
	}

	__CV_END__;

	return (const float**) samples;
}
#endif

/**/ML_IMPL CvCNNStatModel*
cvTrainCNNClassifier( const CvMat* _train_data, int tflag,
            const CvMat* _responses,
            const CvCNNStatModelParams* _params,
            const CvMat*, const CvMat* _sample_idx, const CvMat*, const CvMat* )
{
    CvCNNStatModel* cnn_model    = 0;
    const float** out_train_data = 0;
    const float*** out_train_samples = 0;
    CvMat* responses             = 0;

	CV_FUNCNAME("cvTrainCNNClassifier");
	__CV_BEGIN__;

	int n_images;

	CvCNNStatModelParams* params = (CvCNNStatModelParams*) _params;

    CV_CALL(cnn_model = (CvCNNStatModel*)cvCreateCNNStatModel(
        CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel),
        icvCNNModelRelease, NULL, NULL ));

	CV_CALL( out_train_data =
                cvGetTrainSamples( _train_data, tflag, NULL, NULL,
                                   NULL, &n_images, 1/*always_copy_data*/ )); 

	ICV_CHECK_CNN_MODEL_PARAMS(params);
	ICV_CHECK_CNN_NETWORK(params->network);

	cnn_model->network = params->network;
	CV_CALL(cnn_model->etalons = (CvMat*) cvCloneMat(params->etalons));

    CV_CALL( icvTrainCNNetwork( cnn_model->network, out_train_data, _responses,
        cnn_model->etalons, params->grad_estim_type, params->max_iter,
        params->start_iter ));

	__CV_END__;

	if (cvGetErrStatus() < 0 && cnn_model) {
		cnn_model->release((CvCNNStatModel**) &cnn_model);
	}

	cvFree(&out_train_data);
	cvReleaseMat(&responses);

	return (CvCNNStatModel*) cnn_model;
}

/****************************************************************************************/
static void icvTrainCNNetwork( CvCNNetwork* network,
                               const float** images,
                               const CvMat* responses,
                               const CvMat* etalons,
                               int grad_estim_type,
                               int max_iter,
                               int start_iter )
{
    CvMat** X     = 0;
    CvMat** dE_dX = 0;
    CvMat** d2E_dX2 = 0;
    const int n_layers = network->n_layers;
    int k;

	CV_FUNCNAME("icvTrainCNNetwork");
	__CV_BEGIN__;

	CvCNNLayer* first_layer = network->layers;
	const int img_height = first_layer->input_height;
	const int img_width = first_layer->input_width;
	const int img_size = img_width * img_height;
	const int n_images = responses->rows;
	CvMat image = cvMat(1, img_size, CV_32FC1);
	CvCNNLayer* layer;
	int n;
	CvRNG rng = cvRNG(-1);

	FILE *fplog, *fpprogress;
	static float trloss, sumloss;
	trloss = 0;
	sumloss = 0;

	CvMat* mattemp;
	mattemp = cvCreateMat(etalons->cols, 1, CV_32FC1);
	fplog = fopen("log_mlcnn.txt", "w");
	fpprogress = fopen("training_progress.txt", "w");

	CV_CALL(X = (CvMat**) cvAlloc((n_layers + 1) * sizeof(CvMat*)));
	CV_CALL(dE_dX = (CvMat**) cvAlloc((n_layers + 1) * sizeof(CvMat*)));
	CV_CALL(d2E_dX2 = (CvMat**) cvAlloc((n_layers + 1) * sizeof(CvMat*)));
	memset(X, 0, (n_layers + 1) * sizeof(CvMat*));
	memset(dE_dX, 0, (n_layers + 1) * sizeof(CvMat*));
	memset(d2E_dX2, 0, (n_layers + 1) * sizeof(CvMat*));

    CV_CALL(X[0] = cvCreateMat( img_height*img_width,1,CV_32FC1 ));
    CV_CALL(dE_dX[0] = cvCreateMat( 1, X[0]->rows, CV_32FC1 ));
    CV_CALL(d2E_dX2[0] = cvCreateMat( 1, X[0]->rows, CV_32FC1 ));
    for( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer ){
        CV_CALL(X[k+1] = cvCreateMat( layer->n_output_planes*layer->output_height*layer->output_width, 1, CV_32FC1 ));
        CV_CALL(dE_dX[k+1] = cvCreateMat( 1, X[k+1]->rows, CV_32FC1 ));
        CV_CALL(d2E_dX2[k+1] = cvCreateMat( 1, X[k+1]->rows, CV_32FC1 ));
    }

    for( n = 1; n <= max_iter; n++ ) {

        float loss, max_loss = 0;
        int i;
        int worst_img_idx = -1;
        int* right_etal_idx = responses->data.i;
        CvMat etalon;

		const int numSamples = n_images;

		rewind(fpprogress);
		fprintf(fpprogress,"%d/%d = %f",n,max_iter,float(n)/float(max_iter));
		fflush(fpprogress);
		// Find the worst image (which produces the greatest loss) or use the random image
		if (grad_estim_type == CV_CNN_GRAD_ESTIM_BY_WORST_IMG) {
			for (i = 0; i < n_images; i++, right_etal_idx++) {

				image.data.fl = (float*) images[i];
				cvTranspose(&image, X[0]);

				for (k = 0, layer = first_layer; k < n_layers; k++, layer =	layer->next_layer)
					CV_CALL(layer->forward(layer, X[k], X[k + 1]));

				cvTranspose(X[n_layers], dE_dX[n_layers]);
#ifdef TRAINFACE
				cvGetRow( responses, &etalon, worst_img_idx);
#else
				cvGetRow(etalons, &etalon, *right_etal_idx);
#endif

				loss = (float) cvNorm(dE_dX[n_layers], &etalon);
				if (loss > max_loss) {
					max_loss = loss;
					worst_img_idx = i;
				}
			}
		} else
			worst_img_idx = (n - 1) % numSamples;

		// Train network on the worst image
		// 1) Compute the network output on the <image>
		image.data.fl = (float*) images[worst_img_idx];

		CV_CALL(cvTranspose(&image, X[0]));

        for( k = 0, layer = first_layer; k < n_layers - 1; k++, layer = layer->next_layer )
            CV_CALL(layer->forward( layer, X[k], X[k+1] ));
        CV_CALL(layer->forward( layer, X[k], X[k+1] ));

		// 2) Compute the gradient
		cvTranspose(X[n_layers], dE_dX[n_layers]);
		//cvGetRow(etalon)

#ifdef TRAINFACE
		cvGetRow(responses, &etalon, worst_img_idx);
#else
		cvGetRow(etalons, &etalon, (int) responses->data.fl[worst_img_idx]);
#endif

		cvSub(dE_dX[n_layers], &etalon, dE_dX[n_layers]);

		for (i = 0; i < d2E_dX2[n_layers]->cols; i++)
			d2E_dX2[n_layers]->data.fl[i] = 1.0f;

		// 3) Update weights by the gradient descent
        for( k = n_layers; k > 0; k--, layer = layer->prev_layer )
            CV_CALL(layer->backward( layer, n + start_iter, X[k-1], X[k], dE_dX[k], dE_dX[k-1], d2E_dX2[k], d2E_dX2[k-1] ));

		cvTranspose(&etalon, mattemp);
		trloss = (float) cvNorm(X[n_layers], mattemp);
		sumloss += trloss;
		fflush(fplog);
		fprintf(fplog, "%f\n", sumloss / n);

		if (n % numSamples == 0) {
			FILE *fpweights;
			char *str;
			str = (char*) malloc(sizeof(char) * 100);
			int j;
			CvCNNConvolutionLayer* templayer;

			string chname = "tweights";
			char multNum[3];
			sprintf(multNum, "%d", n / numSamples);
			chname = chname + multNum + ".dat";
			fpweights=fopen(chname.c_str(),"wb");
			templayer=(CvCNNConvolutionLayer*)first_layer;
			for(i=0;i<templayer->weights->rows;i++){
				for(j=0;j<templayer->weights->cols;j++){
					fwrite(&templayer->weights->data.fl[i*templayer->weights->cols+j],sizeof(float),1,fpweights);
				}
			}
			templayer=(CvCNNConvolutionLayer*)first_layer->next_layer;
			for(i=0;i<templayer->weights->rows;i++){
				for(j=0;j<templayer->weights->cols;j++){
					fwrite(&templayer->weights->data.fl[i*templayer->weights->cols+j],sizeof(float),1,fpweights);
				}
			}
			templayer=(CvCNNConvolutionLayer*)first_layer->next_layer->next_layer;
			for(i=0;i<templayer->weights->rows;i++){
				for(j=0;j<templayer->weights->cols;j++)	{
					fwrite(&templayer->weights->data.fl[i*templayer->weights->cols+j],sizeof(float),1,fpweights);
				}
			}
			templayer=(CvCNNConvolutionLayer*)first_layer->next_layer->next_layer->next_layer;
			for(i=0;i<templayer->weights->rows;i++)	{
				for(j=0;j<templayer->weights->cols;j++)	{
					fwrite(&templayer->weights->data.fl[i*templayer->weights->cols+j],sizeof(float),1,fpweights);
				}
			}
			fclose(fpweights);
			free(str);
		}
	}

	fclose(fpprogress);
	fclose(fplog);
	cvReleaseMat(&mattemp);

	__CV_END__;

	for (k = 0; k <= n_layers; k++) {
		cvReleaseMat(&X[k]);
		cvReleaseMat(&dE_dX[k]);
		cvReleaseMat(&d2E_dX2[k]);
	}
	cvFree(&X);
	cvFree(&dE_dX);
	cvFree(&d2E_dX2);
}

CvMat * icvCNNModelPredict(const CvCNNStatModel* model,
		const CvMat* _image, CvMat** output) {
	CvMat** X = 0;
	float* img_data = 0;
	float** out_train_samples = 0;
	int n_layers = 0;
	int best_etal_idx = -1;
	int k;

	CV_FUNCNAME("icvCNNModelPredict");
	__CV_BEGIN__;

	CvCNNStatModel* cnn_model = (CvCNNStatModel*) model;
	CvCNNLayer* first_layer, *layer = 0;
	int img_height, img_width, img_size;
	int nclasses;
	float min_loss = FLT_MAX;
	CvMat image;

	if (!CV_IS_CNN(model))
		CV_ERROR(CV_StsBadArg, "Invalid model");

	nclasses = cnn_model->cls_labels->cols;
	n_layers = cnn_model->network->n_layers;
	first_layer = cnn_model->network->layers;
	img_height = first_layer->input_height;
	img_width = first_layer->input_width;

	CV_CALL(
			out_train_samples = (float **) cvGetTrainSamples(_image,
					CV_ROW_SAMPLE, NULL, NULL, NULL, &img_size,
					1/*always_copy_data*/));

	img_size = img_height * img_width;
	CV_CALL(X = (CvMat**) cvAlloc((n_layers + 1) * sizeof(CvMat*)));
	memset(X, 0, (n_layers + 1) * sizeof(CvMat*));

	CV_CALL(X[0] = cvCreateMat(img_size, 1, CV_32FC1));
	for (k = 0, layer = first_layer; k < n_layers;
			k++, layer = layer->next_layer) {
		CV_CALL(
				X[k + 1] = cvCreateMat(
						layer->n_output_planes * layer->output_height
								* layer->output_width, 1, CV_32FC1));
	}

	image = cvMat(1, img_size, CV_32FC1, *out_train_samples/*img_data*/);
	cvTranspose(&image, X[0]);
	for (k = 0, layer = first_layer; k < n_layers;
			k++, layer = layer->next_layer)
		CV_CALL(layer->forward(layer, X[k], X[k + 1]));

	float *ptr;
	float temp;

	for (int i = 0; i < X[n_layers]->rows; i++) {
		ptr = (float *) (output[n_layers]->data.ptr
				+ i * output[n_layers]->step);
		temp = CV_MAT_ELEM(*X[n_layers],float, i,0);
		*ptr = temp;
	}

	for (k = 0; k <= n_layers; k++)
		cvReleaseMat(&X[k]);
	cvFree(&X);
	if (img_data != _image->data.fl)
		cvFree(&img_data);

	cvFree(&out_train_samples);

	__CV_END__;

	return output[n_layers];
}

ML_IMPL float icvCNNModelPredict( const CvCNNStatModel* model,
                                 const CvMat* _image,
                                 CvMat* probs )
{
    CvMat** X       = 0;
    float* img_data = 0;
    int n_layers = 0;
    int best_etal_idx = -1;
    int k;

    CV_FUNCNAME("icvCNNModelPredict");
    __BEGIN__;

    CvCNNStatModel* cnn_model = (CvCNNStatModel*)model;
    CvCNNLayer* first_layer, *layer = 0;
    int img_height, img_width, img_size;
    int nclasses, i;
    float loss, min_loss = FLT_MAX;
    float* probs_data;
    CvMat etalon, image;

    if( !CV_IS_CNN(model) )
        CV_ERROR( CV_StsBadArg, "Invalid model" );

    nclasses = cnn_model->cls_labels->cols;
    n_layers = cnn_model->network->n_layers;
    first_layer   = cnn_model->network->layers;
    img_height = first_layer->input_height;
    img_width  = first_layer->input_width;
    img_size   = img_height*img_width;

    cvPreparePredictData( _image, img_size, 0, nclasses, probs, &img_data );

    CV_CALL(X = (CvMat**)cvAlloc( (n_layers+1)*sizeof(CvMat*) ));
    memset( X, 0, (n_layers+1)*sizeof(CvMat*) );

    CV_CALL(X[0] = cvCreateMat( img_size,1,CV_32FC1 ));
    for( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer )
    {
        CV_CALL(X[k+1] = cvCreateMat( layer->n_output_planes*layer->output_height*
            layer->output_width, 1, CV_32FC1 ));
    }

    image = cvMat( 1, img_size, CV_32FC1, img_data );
    cvTranspose( &image, X[0] );
    for( k = 0, layer = first_layer; k < n_layers; k++, layer = layer->next_layer )
        CV_CALL(layer->forward( layer, X[k], X[k+1] ));

    probs_data = probs ? probs->data.fl : 0;
    etalon = cvMat( cnn_model->etalons->cols, 1, CV_32FC1, cnn_model->etalons->data.fl );
    for( i = 0; i < nclasses; i++, etalon.data.fl += cnn_model->etalons->cols )
    {
        loss = (float)cvNorm( X[n_layers], &etalon );
        if( loss < min_loss )
        {
            min_loss = loss;
            best_etal_idx = i;
        }
        if( probs )
            *probs_data++ = -loss;
    }

    if( probs )
    {
        cvExp( probs, probs );
        CvScalar sum = cvSum( probs );
        cvConvertScale( probs, probs, 1./sum.val[0] );
    }

    __END__;

    for( k = 0; k <= n_layers; k++ )
        cvReleaseMat( &X[k] );
    cvFree( &X );
    if( img_data != _image->data.fl )
        cvFree( &img_data );

    return ((float) ((CvCNNStatModel*)model)->cls_labels->data.i[best_etal_idx]);
}

void icvCNNModelUpdate(
        CvStatModel* _cnn_model, const CvMat* _train_data, int tflag,
        const CvMat* _responses, const CvStatModelParams* _params,
        const CvMat*, const CvMat* _sample_idx,
        const CvMat*, const CvMat* )
{
    const float** out_train_data = 0;
    CvMat* responses             = 0;
    CvMat* cls_labels            = 0;

    CV_FUNCNAME("icvCNNModelUpdate");
    __BEGIN__;

    int n_images, img_size, i;
    CvCNNStatModelParams* params = (CvCNNStatModelParams*)_params;
    CvCNNStatModel* cnn_model = (CvCNNStatModel*)_cnn_model;

    if( !CV_IS_CNN(cnn_model) )
        CV_ERROR( CV_StsBadArg, "Invalid model" );

    CV_CALL(cvPrepareTrainData( "cvTrainCNNClassifier",
        _train_data, tflag, _responses, CV_VAR_CATEGORICAL,
        0, _sample_idx, false, &out_train_data,
        &n_images, &img_size, &img_size, &responses,
        &cls_labels, 0, 0 ));

    ICV_CHECK_CNN_MODEL_PARAMS(params);

    // Number of classes must be the same as when classifiers was created
    if( !CV_ARE_SIZES_EQ(cls_labels, cnn_model->cls_labels) )
        CV_ERROR( CV_StsBadArg, "Number of classes must be left unchanged" );
    for( i = 0; i < cls_labels->cols; i++ )
    {
        if( cls_labels->data.i[i] != cnn_model->cls_labels->data.i[i] )
            CV_ERROR( CV_StsBadArg, "Number of classes must be left unchanged" );
    }

    CV_CALL( icvTrainCNNetwork( cnn_model->network, out_train_data, responses,
        cnn_model->etalons, params->grad_estim_type, params->max_iter,
        params->start_iter ));

    __END__;

    cvFree( &out_train_data );
    cvReleaseMat( &responses );
}

/****************************************************************************************/
static void icvCNNModelRelease( CvCNNStatModel** cnn_model )
{
    CV_FUNCNAME("icvCNNModelRelease");
    __CV_BEGIN__;

	CvCNNStatModel* cnn;
	if (!cnn_model)
		CV_ERROR(CV_StsNullPtr, "Null double pointer");

	cnn = *(CvCNNStatModel**) cnn_model;

	cvReleaseMat(&cnn->cls_labels);
	cvReleaseMat(&cnn->etalons);
	cnn->network->release(&cnn->network);

	cvFree(&cnn);

	__CV_END__;

}

/****************************************************************************************\
*                                 Network functions                                      *
\****************************************************************************************/
ML_IMPL CvCNNetwork* cvCreateCNNetwork( CvCNNLayer* first_layer )
{
    CvCNNetwork* network = 0;    
    
    CV_FUNCNAME( "cvCreateCNNetwork" );
    __CV_BEGIN__;

	if (!ICV_IS_CNN_LAYER(first_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	CV_CALL(network = (CvCNNetwork*) cvAlloc(sizeof(CvCNNetwork)));
	memset(network, 0, sizeof(CvCNNetwork));

	network->layers = first_layer;
	network->n_layers = 1;
	network->release = icvCNNetworkRelease;
	network->add_layer = icvCNNetworkAddLayer;

	__CV_END__;

	if (cvGetErrStatus() < 0 && network)
		cvFree(&network);

	return network;

}

/****************************************************************************************/
static void icvCNNetworkAddLayer( CvCNNetwork* network, CvCNNLayer* layer )
{
    CV_FUNCNAME( "icvCNNetworkAddLayer" );
    __CV_BEGIN__;

	CvCNNLayer* prev_layer;

	if (network == NULL)
		CV_ERROR(CV_StsNullPtr, "Null <network> pointer");

	prev_layer = network->layers;
	while (prev_layer->next_layer)
		prev_layer = prev_layer->next_layer;

    if( ICV_IS_CNN_FULLCONNECT_LAYER(layer) )
    {
        if( layer->n_input_planes != prev_layer->output_width*prev_layer->output_height*
            prev_layer->n_output_planes )
            CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
        if( layer->input_height != 1 || layer->output_height != 1 ||
            layer->input_width != 1  || layer->output_width != 1 )
            CV_ERROR( CV_StsBadArg, "Invalid size of the new layer" );
    }
    else if( ICV_IS_CNN_CONVOLUTION_LAYER(layer) || ICV_IS_CNN_SUBSAMPLING_LAYER(layer) )
    {
        if( prev_layer->n_output_planes != layer->n_input_planes ||
        prev_layer->output_height   != layer->input_height ||
        prev_layer->output_width    != layer->input_width )
        CV_ERROR( CV_StsBadArg, "Unmatched size of the new layer" );
    }
    else
        CV_ERROR( CV_StsBadArg, "Invalid layer" );

	layer->prev_layer = prev_layer;
	prev_layer->next_layer = layer;
	network->n_layers++;

	__CV_END__;
}

/****************************************************************************************/
static void icvCNNetworkRelease( CvCNNetwork** network_pptr )
{
    CV_FUNCNAME( "icvReleaseCNNetwork" );
    __CV_BEGIN__;

	CvCNNetwork* network = 0;
	CvCNNLayer* layer = 0, *next_layer = 0;
	int k;

	if (network_pptr == NULL)
		CV_ERROR(CV_StsBadArg, "Null double pointer");
	if (*network_pptr == NULL)
		return;

	network = *network_pptr;
	layer = network->layers;
	if (layer == NULL)
		CV_ERROR(CV_StsBadArg, "CNN is empty (does not contain any layer)");

	// k is the number of the layer to be deleted
	for (k = 0; k < network->n_layers && layer; k++) {
		next_layer = layer->next_layer;
		layer->release(&layer);
		layer = next_layer;
	}

	if (k != network->n_layers || layer)
		CV_ERROR(CV_StsBadArg, "Invalid network");

	cvFree(&network);

	__CV_END__;
}

/****************************************************************************************\
*                                  Layer functions                                       *
\****************************************************************************************/
static CvCNNLayer* icvCreateCNNLayer( int layer_type, int header_size,
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int output_height, int output_width,
    float init_learn_rate, int learn_rate_decrease_type, int delta_w_increase_type, int nsamples, int max_iter,
    CvCNNLayerRelease release, CvCNNLayerForward forward, CvCNNLayerBackward backward )
{
    CvCNNLayer* layer = 0;

	CV_FUNCNAME("icvCreateCNNLayer");
	__CV_BEGIN__;

	CV_ASSERT(release && forward && backward)
	CV_ASSERT(header_size >= sizeof(CvCNNLayer))

    if( n_input_planes < 1 || n_output_planes < 1 ||
        input_height   < 1 || input_width < 1 ||
        output_height  < 1 || output_width < 1 ||
        input_height < output_height ||
        input_width  < output_width )
        CV_ERROR( CV_StsBadArg, "Incorrect input or output parameters" );
    if( init_learn_rate < FLT_EPSILON )
        CV_ERROR( CV_StsBadArg, "Initial learning rate must be positive" );
    if( learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_HYPERBOLICALLY &&
        learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_SQRT_INV &&
        learn_rate_decrease_type != CV_CNN_LEARN_RATE_DECREASE_LOG_INV )
        CV_ERROR( CV_StsBadArg, "Invalid type of learning rate dynamics" );

    CV_CALL(layer = (CvCNNLayer*)cvAlloc( header_size ));
    memset( layer, 0, header_size );

	layer->flags = ICV_CNN_LAYER | layer_type;
	CV_ASSERT(ICV_IS_CNN_LAYER(layer))

	layer->n_input_planes = n_input_planes;
	layer->input_height = input_height;
	layer->input_width = input_width;

	layer->n_output_planes = n_output_planes;
	layer->output_height = output_height;
	layer->output_width = output_width;

	layer->init_learn_rate = init_learn_rate;
	layer->learn_rate_decrease_type = learn_rate_decrease_type;
	layer->delta_w_increase_type = delta_w_increase_type;

	layer->release = release;
	layer->forward = forward;
	layer->backward = backward;

	layer->nsamples = nsamples;
	layer->max_iter = max_iter;

	__CV_END__;

	if (cvGetErrStatus() < 0 && layer)
		cvFree(&layer);

	return layer;
}

ML_IMPL CvCNNLayer* cvCreateCNNConvolutionLayer(
    int n_input_planes, int input_height, int input_width,
    int n_output_planes, int K,float a,float s,
    double init_learn_rate, int learn_rate_decrease_type, int delta_w_increase_type, int nsamples, int max_iter,
    CvMat* connect_mask, CvMat* weights )

{
    CvCNNConvolutionLayer* layer = 0;

	CV_FUNCNAME("cvCreateCNNConvolutionLayer");
	__CV_BEGIN__;

	const int output_height = (input_height - 3) / 2;
	const int output_width = (input_width - 3) / 2;

	if (K < 1 || init_learn_rate <= 0)
		CV_ERROR(CV_StsBadArg, "Incorrect parameters");

    CV_CALL(layer = (CvCNNConvolutionLayer*)icvCreateCNNLayer( ICV_CNN_CONVOLUTION_LAYER,
        sizeof(CvCNNConvolutionLayer), n_input_planes, input_height, input_width,
        n_output_planes, output_height, output_width,
        init_learn_rate, learn_rate_decrease_type, delta_w_increase_type,nsamples,max_iter,
        icvCNNConvolutionRelease, icvCNNConvolutionForward, icvCNNConvolutionBackward ));

	layer->K = K;
	layer->a = a;
	layer->s = s;
	CV_CALL( layer->sumX = cvCreateMat( n_output_planes * output_width * output_height, 1, CV_32FC1));
	CV_CALL( layer->exp2ssumWX = cvCreateMat( n_output_planes * output_width * output_height, 1, CV_32FC1));
	cvZero(layer->sumX);
	cvZero(layer->exp2ssumWX);

	CV_CALL( layer->weights = cvCreateMat( n_output_planes * (K * K * n_input_planes + 1), 1, CV_32FC1));
	CV_CALL( layer->connect_mask = cvCreateMat(n_output_planes, n_input_planes, CV_8UC1)); 
	if (weights) {
		if (!ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ))
			CV_ERROR(CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1");
		if (!CV_ARE_SIZES_EQ(weights, layer->weights))
			CV_ERROR(CV_StsBadSize, "Invalid size of initial weights matrix");
		CV_CALL(cvCopy(weights, layer->weights));
	} else {
		CvRNG rng = cvRNG(0xFFFFFFFF);
		int i;
		for (i = 0; i < n_output_planes * (K * K * n_input_planes + 1); i++)
			layer->weights->data.fl[i] = (float) (0.05 * (2.0 * rand() / RAND_MAX - 1.0));
	}

	if (connect_mask) {
		if (!ICV_IS_MAT_OF_TYPE( connect_mask, CV_8UC1 ))
			CV_ERROR(CV_StsBadSize,	"Type of connection matrix must be CV_8UC1");
		if (!CV_ARE_SIZES_EQ(connect_mask, layer->connect_mask))
			CV_ERROR(CV_StsBadSize, "Invalid size of connection matrix");
		CV_CALL(cvCopy(connect_mask, layer->connect_mask));
	} else
		CV_CALL(cvSet(layer->connect_mask, cvRealScalar(1)));

    CV_CALL(layer->hessian_diag = cvCreateMat( layer->weights->rows, layer->weights->cols , CV_32FC1 ));
	cvZero(layer->hessian_diag);

	layer->nsamples = nsamples;
	layer->max_iter = max_iter;

	__CV_END__;

	if (cvGetErrStatus() < 0 && layer) {
		cvReleaseMat(&layer->weights);
		cvReleaseMat(&layer->connect_mask);
		cvReleaseMat(&layer->hessian_diag);
		cvFree(&layer);
	}

	return (CvCNNLayer*) layer;
}

/****************************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNSubSamplingLayer(
    int n_input_planes, int input_height, int input_width,
    int sub_samp_scale, float a, float s,
    float init_learn_rate, int learn_rate_decrease_type,  int delta_w_increase_type, int nsamples, int max_iter,
	CvMat* weights )

{
	CvCNNSubSamplingLayer* layer = 0;

	CV_FUNCNAME("cvCreateCNNSubSamplingLayer");
	__CV_BEGIN__;

	const int output_height = input_height / sub_samp_scale;
	const int output_width = input_width / sub_samp_scale;
	const int n_output_planes = n_input_planes;

	if (sub_samp_scale < 1 || a <= 0 || s <= 0)
		CV_ERROR(CV_StsBadArg, "Incorrect parameters");

    CV_CALL(layer = (CvCNNSubSamplingLayer*)icvCreateCNNLayer( ICV_CNN_SUBSAMPLING_LAYER,
        sizeof(CvCNNSubSamplingLayer), n_input_planes, input_height, input_width,
        n_output_planes, output_height, output_width,
        init_learn_rate, learn_rate_decrease_type, delta_w_increase_type,nsamples,max_iter,
        icvCNNSubSamplingRelease, icvCNNSubSamplingForward, icvCNNSubSamplingBackward ));

	layer->sub_samp_scale = sub_samp_scale;
	layer->a = a;
	layer->s = s;

    CV_CALL(layer->sumX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));
    CV_CALL(layer->exp2ssumWX =
        cvCreateMat( n_output_planes*output_width*output_height, 1, CV_32FC1 ));

	cvZero(layer->sumX);
	cvZero(layer->exp2ssumWX);

	CV_CALL(layer->weights = cvCreateMat(n_output_planes * 2, 1, CV_32FC1));
	if (weights) {
		if (!ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ))
			CV_ERROR(CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1");
		if (!CV_ARE_SIZES_EQ(weights, layer->weights))
			CV_ERROR(CV_StsBadSize, "Invalid size of initial weights matrix");
		CV_CALL(cvCopy(weights, layer->weights));
	} else {
		CvRNG rng = cvRNG(0xFFFFFFFF);
		cvRandArr(&rng, layer->weights, CV_RAND_UNI, cvRealScalar(-1), cvRealScalar(1));
	}

	CV_CALL( layer->hessian_diag = cvCreateMat(layer->weights->rows, layer->weights->cols, CV_32FC1));
	cvZero(layer->hessian_diag);

	layer->nsamples = nsamples;
	layer->max_iter = max_iter;

	__CV_END__;

	if (cvGetErrStatus() < 0 && layer) {
		cvReleaseMat(&layer->exp2ssumWX);
		cvFree(&layer);
	}

	return (CvCNNLayer*) layer;
}

/****************************************************************************************/
ML_IMPL CvCNNLayer* cvCreateCNNFullConnectLayer(int n_inputs, int n_outputs,
		float a, float s, float init_learn_rate, int learn_rate_decrease_type,
		int delta_w_increase_type, int nsamples, int max_iter, CvMat* weights) 
{
	CvCNNFullConnectLayer* layer = 0;

	CV_FUNCNAME("cvCreateCNNFullConnectLayer");
	__CV_BEGIN__;

	if (a <= 0 || s <= 0 || init_learn_rate <= 0)
		CV_ERROR(CV_StsBadArg, "Incorrect parameters");

    CV_CALL(layer = (CvCNNFullConnectLayer*)icvCreateCNNLayer( ICV_CNN_FULLCONNECT_LAYER,
        sizeof(CvCNNFullConnectLayer), n_inputs, 1, 1, n_outputs, 1, 1,
        init_learn_rate, learn_rate_decrease_type, delta_w_increase_type, nsamples,  max_iter, 
        icvCNNFullConnectRelease, icvCNNFullConnectForward, icvCNNFullConnectBackward ));

	layer->a = a;
	layer->s = s;

	CV_CALL(layer->exp2ssumWX = cvCreateMat(n_outputs, 1, CV_32FC1));
	cvZero(layer->exp2ssumWX);

	CV_CALL(layer->weights = cvCreateMat(n_outputs, n_inputs + 1, CV_32FC1));
	if (weights) {
		if (!ICV_IS_MAT_OF_TYPE( weights, CV_32FC1 ))
			CV_ERROR(CV_StsBadSize, "Type of initial weights matrix must be CV_32FC1");
		if (!CV_ARE_SIZES_EQ(weights, layer->weights))
			CV_ERROR(CV_StsBadSize, "Invalid size of initial weights matrix");
		CV_CALL(cvCopy(weights, layer->weights));
	} else {
		CvRNG rng = cvRNG(0xFFFFFFFF);
		int i;
		for (i = 0; i < layer->weights->rows * layer->weights->cols; i++)
			layer->weights->data.fl[i] = (float) (0.05 * (2.0 * rand() / RAND_MAX - 1.0));
	}

	CV_CALL( layer->hessian_diag = cvCreateMat(layer->weights->rows, layer->weights->cols, CV_32FC1));
	cvZero(layer->hessian_diag);

	layer->nsamples = nsamples;
	layer->max_iter = max_iter;

	__CV_END__;

	if (cvGetErrStatus() < 0 && layer) {
		cvReleaseMat(&layer->exp2ssumWX);
		cvReleaseMat(&layer->weights);
		cvReleaseMat(&layer->hessian_diag);
		cvFree(&layer);
	}

	return (CvCNNLayer*) layer;
}

/****************************************************************************************\
*                           Layer FORWARD functions                                      *
\****************************************************************************************/
static void icvCNNConvolutionForward( CvCNNLayer* _layer,
                                      const CvMat* X,
                                      CvMat* Y )
{
    CV_FUNCNAME("icvCNNConvolutionForward");

	if (!ICV_IS_CNN_CONVOLUTION_LAYER(_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	{
		__CV_BEGIN__;

		const CvCNNConvolutionLayer* layer = (CvCNNConvolutionLayer*) _layer;

		const int K = layer->K;
		const int n_weights_for_Yplane = K * K * layer->n_input_planes + 1;

		const int nXplanes = layer->n_input_planes;
		const int Xheight = layer->input_height;
		const int Xwidth = layer->input_width;
		const int Xsize = Xwidth * Xheight;

		const int nYplanes = layer->n_output_planes;
		const int Yheight = layer->output_height;
		const int Ywidth = layer->output_width;
		const int Ysize = Ywidth * Yheight;

		int ni, no, Yi, Yj, Xi, Xj;
		float *pY = 0, *pX = 0, *w = 0;
		uchar* connect_mask_data = 0;

		cvSetZero(Y);

		pY = Y->data.fl;
		pX = X->data.fl;
		connect_mask_data = layer->connect_mask->data.ptr;
		w = layer->weights->data.fl;

		for (no = 0; no < nYplanes; no++) {
			for (Yi = 0; Yi < Yheight; Yi++) {
				for (Yj = 0; Yj < Ywidth; Yj++) {
					for (ni = 0; ni < nXplanes; ni++) {
						for (Xi = 0; Xi < K; Xi++) {
							for (Xj = 0; Xj < K; Xj++) {
								pY[no * Ysize + Yi * Ywidth + Yj] += 
									w[no * (nXplanes * K * K + 1) + ni * K * K + Xi * K + Xj]
									* pX[ni * Xsize + (Yi * 2 + Xi) * Xwidth + (Yj * 2 + Xj)];
							}
						}
					}
					pY[no * Ysize + Yi * Ywidth + Yj] += w[no * (nXplanes * K * K + 1) + nXplanes * K * K];
					pY[no * Ysize + Yi * Ywidth + Yj] = SIG(pY[no*Ysize + Yi*Ywidth + Yj]);
				}
			}
		}

	}
	__CV_END__;
}

/****************************************************************************************/
static void icvCNNSubSamplingForward( CvCNNLayer* _layer,
                                      const CvMat* X,
                                      CvMat* Y )
{
    CV_FUNCNAME("icvCNNSubSamplingForward");

	if (!ICV_IS_CNN_SUBSAMPLING_LAYER(_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	{
		__CV_BEGIN__;

		const CvCNNSubSamplingLayer* layer = (CvCNNSubSamplingLayer*) _layer;

		const int sub_sampl_scale = layer->sub_samp_scale;
		const int nplanes = layer->n_input_planes;

		const int Xheight = layer->input_height;
		const int Xwidth = layer->input_width;
		const int Xsize = Xwidth * Xheight;

		const int Yheight = layer->output_height;
		const int Ywidth = layer->output_width;
		const int Ysize = Ywidth * Yheight;

		int Yi, Yj, Xi, Xj, no;
		float* sumX_data = 0, *w = 0;
		float *pY, *pX;

		CV_ASSERT(X->rows == nplanes * Xsize && X->cols == 1);
		CV_ASSERT(
				layer->exp2ssumWX->cols == 1
						&& layer->exp2ssumWX->rows == nplanes * Ysize);

		// update inner variable layer->exp2ssumWX, which will be used in back-progation
		cvSetZero(Y);

		pY = Y->data.fl;
		pX = X->data.fl;
		w = layer->weights->data.fl;

		for (no = 0; no < nplanes; no++) {
			for (Yi = 0; Yi < Yheight; Yi++) {
				for (Yj = 0; Yj < Ywidth; Yj++) {
					for (Xi = 0; Xi < sub_sampl_scale; Xi++) {
						for (Xj = 0; Xj < sub_sampl_scale; Xj++) {
							pY[no * Ysize + Yi * Ywidth + Yj] += 
								pX[no * Xsize + (Yi * 2 + Xi) * Xwidth + (Yj * 2 + Xj)];
						}
					}
					pY[no * Ysize + Yi * Ywidth + Yj] = 
						pY[no * Ysize + Yi * Ywidth + Yj] * w[no * 2] + w[no * 2 + 1];
					pY[no * Ysize + Yi * Ywidth + Yj] =
						SIG(pY[no*Ysize + Yi*Ywidth + Yj]);
				}
			}
		}

	}
	__CV_END__;
}

static void icvCNNFullConnectForward(CvCNNLayer* _layer, const CvMat* X, CvMat* Y) 
{
	CV_FUNCNAME("icvCNNFullConnectForward");

	if (!ICV_IS_CNN_FULLCONNECT_LAYER(_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	{
		__CV_BEGIN__;

		const CvCNNFullConnectLayer* layer = (CvCNNFullConnectLayer*) _layer;
		CvMat* weights = layer->weights;
		CvMat sub_weights, bias;
		int i;

		CV_ASSERT(X->cols == 1 && X->rows == layer->n_input_planes);
		CV_ASSERT(Y->cols == 1 && Y->rows == layer->n_output_planes);

		CV_CALL( cvGetSubRect(weights, &sub_weights, cvRect(0, 0, weights->cols - 1, weights->rows)));
		CV_CALL(cvGetCol(weights, &bias, weights->cols - 1));

		//(1.7159*tanh(0.66666667*x))
		CV_CALL( cvGEMM(&sub_weights, X, 1/*2*layer->s*/, &bias, 1/*2*layer->s*/, layer->exp2ssumWX));
		for (i = 0; i < Y->rows; i++) {
			Y->data.fl[i] = SIG( layer->exp2ssumWX->data.fl[ i ] );
		}

	}
	__CV_END__;
}

/****************************************************************************************\
*                           Layer BACKWARD functions                                     *
 \****************************************************************************************/

/* <dE_dY>, <dE_dX> should be row-vectors.
 Function computes partial derivatives <dE_dX>
 of the loss function with respect to the planes components
 of the previous layer (X).
 It is a basic function for back propagation method.
 Input parameter <dE_dY> is the partial derivative of the
 loss function with respect to the planes components
 of the current layer. */
static void icvCNNConvolutionBackward(CvCNNLayer* _layer, int t, const CvMat* X,
		CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2, CvMat* d2E_dX2)

{
	CvMat* dY_dX = 0;
	CvMat* dY_dW = 0;
	CvMat* dE_dW = 0;
	CvMat* dErr_dYn_Tmp = 0;
	int ii;
	float output;

	CV_FUNCNAME("icvCNNConvolutionBackward");

	if (!ICV_IS_CNN_CONVOLUTION_LAYER(_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	{
		__CV_BEGIN__;

		CvCNNConvolutionLayer* layer = (CvCNNConvolutionLayer*) _layer;

		const int K = layer->K;

		const int n_X_planes = layer->n_input_planes;
		const int X_plane_height = layer->input_height;
		const int X_plane_width = layer->input_width;
		const int X_plane_size = X_plane_height * X_plane_width;

		const int n_Out_planes = layer->n_output_planes;
		const int Y_plane_height = layer->output_height;
		const int Y_plane_width = layer->output_width;
		const int Y_plane_size = Y_plane_height * Y_plane_width;

		int no, ni;
		int X_idx = 0, Y_idx = 0;

		float *X_plane = 0, *w = 0;

		CvMat* weights = layer->weights;

		CV_ASSERT(t >= 1);

		dY_dW = cvCreateMat(n_Out_planes * (n_X_planes * K * K + 1), 1, CV_32FC1);
		dE_dW = cvCreateMat(n_Out_planes * (n_X_planes * K * K + 1), 1, CV_32FC1);
		dErr_dYn_Tmp = cvCreateMat(Y_plane_size * n_Out_planes, 1, CV_32FC1);

		cvZero(dY_dW);

		for (ii = 0; ii < Y_plane_size * n_Out_planes; ++ii) {
			output = Y->data.fl[ii];

			dErr_dYn_Tmp->data.fl[ii] = DSIG( output ) * dE_dY->data.fl[ii];
		}

		int Yi, Yj, Xi, Xj;
		float *pY = 0, *pX = 0;
		cvZero(dE_dW);
		pY = dErr_dYn_Tmp->data.fl;
		pX = X->data.fl;
		for (no = 0; no < n_Out_planes; no++) {
			for (Yi = 0; Yi < Y_plane_height; Yi++) {
				for (Yj = 0; Yj < Y_plane_width; Yj++) {
					for (ni = 0; ni < n_X_planes; ni++) {
						for (Xi = 0; Xi < K; Xi++) {
							for (Xj = 0; Xj < K; Xj++) {
								output = pX[ni * X_plane_size + (Yi * 2 + Xi) * X_plane_width + (Yj * 2 + Xj)];
								dE_dW->data.fl[no * (n_X_planes * K * K + 1) + ni * K * K + Xi * K + Xj] += output
										* pY[no * Y_plane_size + Yi * Y_plane_width + Yj];
							}
						}
					}
					output = 1;
					dE_dW->data.fl[no * (n_X_planes * K * K + 1) + n_X_planes * K * K] += output
							* pY[no * Y_plane_size + Yi * Y_plane_width + Yj];
				}
			}
		}

		cvZero(dE_dX);
		pY = dErr_dYn_Tmp->data.fl;
		pX = dE_dX->data.fl;
		w = weights->data.fl;
		for (no = 0; no < n_Out_planes; no++) {
			for (Yi = 0; Yi < Y_plane_height; Yi++) {
				for (Yj = 0; Yj < Y_plane_width; Yj++) {
					for (ni = 0; ni < n_X_planes; ni++) {
						for (Xi = 0; Xi < K; Xi++) {
							for (Xj = 0; Xj < K; Xj++) {
								pX[ni * X_plane_size + (Yi * 2 + Xi) * X_plane_width + (Yj * 2 + Xj)] += pY[no * Y_plane_size
										+ Yi * Y_plane_width + Yj]
										* w[no * (n_X_planes * 25 + 1) + ni * 25 + Xi * 5 + Xj];
							}
						}
					}
				}
			}
		}

		if (layer->delta_w_increase_type == CV_CNN_DELTA_W_INCREASE_LM)
		{
			// 1. A=J_t*J; w=w+eta*A_inv*dE_dW
			{
				CvMat *d2E_dW2, *d2Err_dYn_Tmp2;

				d2E_dW2 = cvCreateMat(n_Out_planes * (n_X_planes * K * K + 1), 1, CV_32FC1);
				d2Err_dYn_Tmp2 = cvCreateMat(Y_plane_size * n_Out_planes, 1, CV_32FC1);

				cvZero(d2Err_dYn_Tmp2);
				cvZero(d2E_dW2);
				cvZero(d2E_dX2);

				for (ii = 0; ii < Y_plane_size * n_Out_planes; ++ii) {
					output = Y->data.fl[ii];

					d2Err_dYn_Tmp2->data.fl[ii] = DSIG( output ) * DSIG( output ) * d2E_dY2->data.fl[ii];
				}

				int Yi, Yj, Xi, Xj;
				float *pY = 0, *pX = 0;
				pY = d2Err_dYn_Tmp2->data.fl;
				pX = X->data.fl;
				for (no = 0; no < n_Out_planes; no++) {
					for (Yi = 0; Yi < Y_plane_height; Yi++) {
						for (Yj = 0; Yj < Y_plane_width; Yj++) {
							for (ni = 0; ni < n_X_planes; ni++) {
								for (Xi = 0; Xi < K; Xi++) {
									for (Xj = 0; Xj < K; Xj++) {
										output = pX[ni * X_plane_size + (Yi * 2 + Xi) * X_plane_width + (Yj * 2 + Xj)];
										d2E_dW2->data.fl[no * (n_X_planes * K * K + 1) + ni * K * K + Xi * K + Xj] +=
												output * output * pY[no * Y_plane_size + Yi * Y_plane_width + Yj];
									}
								}
							}
							output = 1;
							d2E_dW2->data.fl[no * (n_X_planes * K * K + 1) + n_X_planes * K * K] += output * output
									* pY[no * Y_plane_size + Yi * Y_plane_width + Yj];
						}
					}
				}

				pY = d2Err_dYn_Tmp2->data.fl;
				pX = d2E_dX2->data.fl;
				w = weights->data.fl;
				for (no = 0; no < n_Out_planes; no++) {
					for (Yi = 0; Yi < Y_plane_height; Yi++) {
						for (Yj = 0; Yj < Y_plane_width; Yj++) {
							for (ni = 0; ni < n_X_planes; ni++) {
								for (Xi = 0; Xi < K; Xi++) {
									for (Xj = 0; Xj < K; Xj++) {
										pX[ni * X_plane_size + (Yi * 2 + Xi) * X_plane_width + (Yj * 2 + Xj)] +=
												pY[no * Y_plane_size + Yi * Y_plane_width + Yj]
														* w[no * (n_X_planes * 25 + 1) + ni * 25 + Xi * 5 + Xj]
														* w[no * (n_X_planes * 25 + 1) + ni * 25 + Xi * 5 + Xj];
									}
								}
							}
						}
					}
				}

				float eta;
				if (layer->learn_rate_decrease_type	== CV_CNN_LEARN_RATE_DECREASE_LOG_INV){
					eta = -layer->init_learn_rate / logf(1 + (float) t);
				}else if (layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV){
					eta = -layer->init_learn_rate / sqrtf( sqrtf((float) ( (t % layer->nsamples) ? (t%layer->nsamples):(1))));
				}else{
					eta = -layer->init_learn_rate/(float)t;
				}

				if ((t + 1) % layer->nsamples == 0)
					layer->init_learn_rate /= 1.08;

				for (ii = 0; ii < weights->cols * weights->rows; ii++) {
					weights->data.fl[ii] = weights->data.fl[ii]
							+ eta / (d2E_dW2->data.fl[ii] + 1.0f)
									* dE_dW->data.fl[ii];
				}

				cvReleaseMat(&d2E_dW2);
				cvReleaseMat(&d2Err_dYn_Tmp2);
			}
		} else
		// update weights
		{
			float eta;
			if (layer->learn_rate_decrease_type
					== CV_CNN_LEARN_RATE_DECREASE_LOG_INV){
				eta = -layer->init_learn_rate / logf(1 + (float) t);
			}else if (layer->learn_rate_decrease_type
					== CV_CNN_LEARN_RATE_DECREASE_SQRT_INV){
				eta = -layer->init_learn_rate / sqrtf( sqrtf((float) ((t % layer->nsamples) ? (t%layer->nsamples):(1))));
			}else{
				eta = -layer->init_learn_rate/(float)t;
			}

			if ((t + 1) % layer->nsamples == 0)
				layer->init_learn_rate /= 1.08;
			cvScaleAdd(dE_dW, cvRealScalar(eta), weights, weights);
		}

	}
	__CV_END__;

	cvReleaseMat(&dY_dW);
	cvReleaseMat(&dE_dW);
	cvReleaseMat(&dErr_dYn_Tmp);

}

/****************************************************************************************/
static void icvCNNSubSamplingBackward(CvCNNLayer* _layer, int t, const CvMat* X,
		CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2,
		CvMat* d2E_dX2) {
	// derivative of active function
	CvMat* dY_dX_elems = 0; // elements of matrix dY_dX
	CvMat* dY_dW_elems = 0; // elements of matrix dY_dW
	CvMat* dE_dW = 0;
	CvMat* dY_dW = 0;
	CvMat* dErr_dYn_Tmp = 0;
	int ii;
	float output;

	CV_FUNCNAME("icvCNNSubSamplingBackward");

	if (!ICV_IS_CNN_SUBSAMPLING_LAYER(_layer)) {
		CV_ERROR(CV_StsBadArg, "Invalid layer");
	}

	{
		__CV_BEGIN__;

		CvCNNSubSamplingLayer* layer = (CvCNNSubSamplingLayer*) _layer;

		const int Xwidth = layer->input_width;
		const int Xheight = layer->input_height;
		const int Xsize = Xwidth * Xheight;
		const int Ywidth = layer->output_width;
		const int Yheight = layer->output_height;
		const int Ysize = Ywidth * Yheight;
		const int scale = layer->sub_samp_scale;
		const int k_max = layer->n_output_planes * Yheight;
		const int n_Out_planes = layer->n_output_planes;


    	float* dY_dX_current_elem = 0, *dE_dX_start = 0, *dE_dW_data = 0, *w = 0;


		CvMat* weights = layer->weights;

		CV_ASSERT(t >= 1);

		dY_dW = cvCreateMat(n_Out_planes * 2, 1, CV_32FC1);
		dE_dW = cvCreateMat(n_Out_planes * 2, 1, CV_32FC1);
		dErr_dYn_Tmp = cvCreateMat(Ysize * n_Out_planes, 1, CV_32FC1);

		cvZero(dY_dW);

		for (ii = 0; ii < Ysize * n_Out_planes; ii++) {
			output = Y->data.fl[ii];

			dErr_dYn_Tmp->data.fl[ii] = DSIG( output ) * dE_dY->data.fl[ii];
		}

		int Yi, Yj, Xi, Xj, no;
		float *pY = 0, *pX = 0;
		cvZero(dE_dW);
		pY = dErr_dYn_Tmp->data.fl;
		pX = X->data.fl;
		for (no = 0; no < n_Out_planes; no++) {
			for (Yi = 0; Yi < Yheight; Yi++) {
				for (Yj = 0; Yj < Ywidth; Yj++) {
					output = 0;
					for (Xi = 0; Xi < scale; Xi++) {
						for (Xj = 0; Xj < scale; Xj++) {
							output += pX[no * Xsize + (Yi * scale + Xi) * Xwidth + (Yj * scale + Xj)];
						}
					}
					dE_dW->data.fl[no * 2] += output
							* pY[no * Ysize + Yi * Ywidth + Yj];
					output = 1;
					dE_dW->data.fl[no * 2 + 1] += output
							* pY[no * Ysize + Yi * Ywidth + Yj];
				}
			}
		}

		cvZero(dE_dX);
		pY = dErr_dYn_Tmp->data.fl;
		pX = dE_dX->data.fl;
		w = weights->data.fl;
		for (no = 0; no < n_Out_planes; no++) {
			for (Yi = 0; Yi < Yheight; Yi++) {
				for (Yj = 0; Yj < Ywidth; Yj++) {
					for (Xi = 0; Xi < scale; Xi++) {
						for (Xj = 0; Xj < scale; Xj++) {
							pX[no * Xsize + (Yi * 2 + Xi) * Xwidth + (Yj * 2 + Xj)] += pY[no * Ysize + Yi * Ywidth + Yj] * w[no * scale];
						}
					}
				}
			}
		}

		if (layer->delta_w_increase_type == CV_CNN_DELTA_W_INCREASE_LM)
		{
			CvMat *d2E_dW2, *d2Err_dYn_Tmp2;

			d2E_dW2 = cvCreateMat(n_Out_planes * 2, 1, CV_32FC1);
			d2Err_dYn_Tmp2 = cvCreateMat(n_Out_planes * 2, 1, CV_32FC1);

			cvZero(d2Err_dYn_Tmp2);
			cvZero(d2E_dW2);
			cvZero(d2E_dX2);

			for (ii = 0; ii < Ysize * n_Out_planes; ii++) {
				output = Y->data.fl[ii];

				d2Err_dYn_Tmp2->data.fl[ii] = DSIG( output )
						* DSIG( output ) * d2E_dY2->data.fl[ii];
			}

			int Yi, Yj, Xi, Xj, no;
			float *pY = 0, *pX = 0;
			cvZero(dE_dW);
			pY = d2Err_dYn_Tmp2->data.fl;
			pX = X->data.fl;
			for (no = 0; no < n_Out_planes; no++) {
				for (Yi = 0; Yi < Yheight; Yi++) {
					for (Yj = 0; Yj < Ywidth; Yj++) {
						output = 0;
						for (Xi = 0; Xi < scale; Xi++) {
							for (Xj = 0; Xj < scale; Xj++) {
								output += pX[no * Xsize + (Yi * scale + Xi) * Xwidth + (Yj * scale + Xj)];
							}
						}
						d2E_dW2->data.fl[no * 2] += output * output * pY[no * Ysize + Yi * Ywidth + Yj];
						output = 1;
						d2E_dW2->data.fl[no * 2 + 1] += output * output * pY[no * Ysize + Yi * Ywidth + Yj];
					}
				}
			}

			cvZero(dE_dX);
			pY = d2Err_dYn_Tmp2->data.fl;
			pX = d2E_dX2->data.fl;
			w = weights->data.fl;
			for (no = 0; no < n_Out_planes; no++) {
				for (Yi = 0; Yi < Yheight; Yi++) {
					for (Yj = 0; Yj < Ywidth; Yj++) {
						for (Xi = 0; Xi < scale; Xi++) {
							for (Xj = 0; Xj < scale; Xj++) {
								pX[no * Xsize + (Yi * 2 + Xi) * Xwidth + (Yj * 2 + Xj)] += pY[no * Ysize + Yi * Ywidth + Yj] 
									* w[no * scale]
									* w[no * scale];
							}
						}
					}
				}
			}

			float eta;
			if (layer->learn_rate_decrease_type	== CV_CNN_LEARN_RATE_DECREASE_LOG_INV){
				eta = -layer->init_learn_rate / logf(1 + (float) t);
			}else if (layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV){
				eta = -layer->init_learn_rate / sqrtf( sqrtf( (float) ((t % layer->nsamples) ? (t%layer->nsamples):(1))));
			}else{
				eta = -layer->init_learn_rate/(float)t;
			}

			if ((t + 1) % layer->nsamples == 0) {
				layer->init_learn_rate /= 1.08;
			}

			for (ii = 0; ii < weights->cols * weights->rows; ii++) {
				weights->data.fl[ii] = weights->data.fl[ii]
						+ eta / (d2E_dW2->data.fl[ii] + 1.0f)
								* dE_dW->data.fl[ii];
			}

			cvReleaseMat(&d2E_dW2);
			cvReleaseMat(&d2Err_dYn_Tmp2);
		} else { // update weights
			float eta;
			if (layer->learn_rate_decrease_type	== CV_CNN_LEARN_RATE_DECREASE_LOG_INV) {
				eta = -layer->init_learn_rate / logf(1 + (float) t);
			} else if (layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV) {
				eta = -layer->init_learn_rate / sqrtf( sqrtf( (float) (	(t % layer->nsamples) ? (t%layer->nsamples):(1))));
			} else {
				eta = -layer->init_learn_rate/(float)t;
			}

			if ((t + 1) % layer->nsamples == 0)
				layer->init_learn_rate /= 1.08;

			cvScaleAdd(dE_dW, cvRealScalar(eta), weights, weights);
		}

	}
	__CV_END__;

	cvReleaseMat(&dY_dW);
	cvReleaseMat(&dE_dW);
	cvReleaseMat(&dErr_dYn_Tmp);
}

/****************************************************************************************/
/* <dE_dY>, <dE_dX> should be row-vectors.
 Function computes partial derivatives <dE_dX>, <dE_dW>
 of the loss function with respect to the planes components
 of the previous layer (X) and the weights of the current layer (W)
 and updates weights od the current layer by using <dE_dW>.
 It is a basic function for back propagation method.
 Input parameter <dE_dY> is the partial derivative of the
 loss function with respect to the planes components
 of the current layer. */
static void icvCNNFullConnectBackward(CvCNNLayer* _layer, int t, const CvMat* X,
		CvMat* Y, const CvMat* dE_dY, CvMat* dE_dX, const CvMat* d2E_dY2,
		CvMat* d2E_dX2) 
{
	CvMat* dE_dY_af = 0;
	CvMat* dE_dW = 0;
	CvMat* dErr_dYn_Tmp = 0;
	float output;
	int ii;

	CV_FUNCNAME("icvCNNFullConnectBackward");

	if (!ICV_IS_CNN_FULLCONNECT_LAYER(_layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	{
		__CV_BEGIN__;

		CvCNNFullConnectLayer* layer = (CvCNNFullConnectLayer*) _layer;
		const int n_outputs = layer->n_output_planes;
		const int n_inputs = layer->n_input_planes;

		int i, j;
		CvMat* weights = layer->weights;

		CV_ASSERT(X->cols == 1 && X->rows == n_inputs);
		CV_ASSERT(dE_dY->rows == 1 && dE_dY->cols == n_outputs);
		CV_ASSERT(dE_dX->rows == 1 && dE_dX->cols == n_inputs);

		// we violate the convetion about vector's orientation because
		// here is more convenient to make this parameter a row-vector
		// dE_dY_af: the active function
		CV_CALL(dE_dY_af = cvCreateMat(1, n_outputs, CV_32FC1));
		CV_CALL(dE_dW = cvCreateMat(1, weights->rows * weights->cols,CV_32FC1));

		for (ii = 0; ii < n_outputs; ++ii) {
			output = Y->data.fl[ii];

			dE_dY_af->data.fl[ii] = DSIG( output ) * dE_dY->data.fl[ii];
		}

		cvZero(dE_dW);
		for (i = 0; i < n_outputs; i++) {
			for (j = 0; j < n_inputs; j++) {
				// dY_dWi, i=1,...,K*K
				output = X->data.fl[j];
				dE_dW->data.fl[i * (n_inputs + 1) + j] += output * dE_dY_af->data.fl[i];
			}
			output = 1;
			dE_dW->data.fl[i * (n_inputs + 1) + n_inputs] += output
					* dE_dY_af->data.fl[i];
		}

		cvZero(dE_dX);
		for (j = 0; j < n_outputs; j++) {
			for (i = 0; i < n_inputs; i++) {
				dE_dX->data.fl[i] += dE_dY_af->data.fl[j]
						* weights->data.fl[j * (n_inputs + 1) + i];
			}
		}

		if (layer->delta_w_increase_type == CV_CNN_DELTA_W_INCREASE_LM)
		{
			// 1. A=J_t*J; w=w+eta*A_inv*dE_dW
			{
				CvMat *d2E_dY_af2, *d2E_dW2;
				CV_CALL(
						d2E_dY_af2 = cvCreateMat(1, n_outputs, CV_32FC1));
				CV_CALL(
						d2E_dW2 = cvCreateMat(1, weights->rows * weights->cols, CV_32FC1));

				for (ii = 0; ii < n_outputs; ++ii) {
					output = Y->data.fl[ii];

					d2E_dY_af2->data.fl[ii] = DSIG( output ) * DSIG( output ) * d2E_dY2->data.fl[ii];
				}

				cvZero(d2E_dW2);
				for (i = 0; i < n_outputs; i++) {
					for (j = 0; j < n_inputs; j++) {
						// dY_dWi, i=1,...,K*K
						output = X->data.fl[j];
						d2E_dW2->data.fl[i * (n_inputs + 1) + j] += output
								* output * d2E_dY_af2->data.fl[i];
					}
					output = 1;
					d2E_dW2->data.fl[i * (n_inputs + 1) + n_inputs] += output
							* output * d2E_dY_af2->data.fl[i];
				}

				cvZero(d2E_dX2);
				for (j = 0; j < n_outputs; j++) {
					for (i = 0; i < n_inputs; i++) {
						d2E_dX2->data.fl[i] +=
								d2E_dY_af2->data.fl[j]
										* weights->data.fl[j * (n_inputs + 1)
												+ i]
										* weights->data.fl[j * (n_inputs + 1)
												+ i];
					}
				}

				float eta;
				if (layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_LOG_INV) {
					eta = -layer->init_learn_rate / logf(1 + (float) t);
				}else if (layer->learn_rate_decrease_type == CV_CNN_LEARN_RATE_DECREASE_SQRT_INV){
					eta = -layer->init_learn_rate / sqrtf( sqrtf( (float) (	(t % layer->nsamples) ? (t%layer->nsamples):(1))));
				}else{
					eta = -layer->init_learn_rate/(float)t;
				}

				if ((t + 1) % layer->nsamples == 0)
					layer->init_learn_rate /= 1.08;

				for (ii = 0; ii < weights->cols * weights->rows; ii++) {
					weights->data.fl[ii] = weights->data.fl[ii]	+ eta / (d2E_dW2->data.fl[ii] + 1.0f)
									* dE_dW->data.fl[ii];
				}

				cvReleaseMat(&d2E_dW2);
				cvReleaseMat(&d2E_dY_af2);
			}
		} else {// 2) update weights
			CvMat dE_dW_mat;
			float eta;
			if (layer->learn_rate_decrease_type
					== CV_CNN_LEARN_RATE_DECREASE_LOG_INV){
				eta = -layer->init_learn_rate / logf(1 + (float) t);
			}else if (layer->learn_rate_decrease_type
					== CV_CNN_LEARN_RATE_DECREASE_SQRT_INV){
				eta = -layer->init_learn_rate / sqrtf( sqrtf( (float) ( (t % layer->nsamples) ? (t%layer->nsamples):(1))));
			}else{
				eta = -layer->init_learn_rate/(float)t;
			}

			if ((t + 1) % layer->nsamples == 0){
				layer->init_learn_rate /= 1.08;
			}

			cvReshape(dE_dW, &dE_dW_mat, 0, n_outputs);
			cvScaleAdd(&dE_dW_mat, cvRealScalar(eta), weights, weights);
		}

	}
	__CV_END__;

	cvReleaseMat(&dE_dY_af);
	cvReleaseMat(&dE_dW);
}

/****************************************************************************************\
*                           Layer RELEASE functions                                      *
\****************************************************************************************/
static void icvCNNConvolutionRelease( CvCNNLayer** p_layer )
{
    CV_FUNCNAME("icvCNNConvolutionRelease");
    __CV_BEGIN__;

	CvCNNConvolutionLayer* layer = 0;

	if (!p_layer)
		CV_ERROR(CV_StsNullPtr, "Null double pointer");

	layer = *(CvCNNConvolutionLayer**) p_layer;

	if (!layer)
		return;
	if (!ICV_IS_CNN_CONVOLUTION_LAYER(layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	cvReleaseMat(&layer->weights);
	if (layer->hessian_diag)
		cvReleaseMat(&layer->hessian_diag);
	cvReleaseMat(&layer->connect_mask);
	cvFree(p_layer);

	__CV_END__;
}

/****************************************************************************************/
static void icvCNNSubSamplingRelease( CvCNNLayer** p_layer )
{
    CV_FUNCNAME("icvCNNSubSamplingRelease");
    __CV_BEGIN__;

	CvCNNSubSamplingLayer* layer = 0;

	if (!p_layer)
		CV_ERROR(CV_StsNullPtr, "Null double pointer");

	layer = *(CvCNNSubSamplingLayer**) p_layer;

	if (!layer)
		return;
	if (!ICV_IS_CNN_SUBSAMPLING_LAYER(layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	cvReleaseMat(&layer->exp2ssumWX);
	cvReleaseMat(&layer->weights);
	cvFree(p_layer);

	__CV_END__;
}

/****************************************************************************************/
static void icvCNNFullConnectRelease( CvCNNLayer** p_layer )
{
    CV_FUNCNAME("icvCNNFullConnectRelease");
    __CV_BEGIN__;

	CvCNNFullConnectLayer* layer = 0;

	if (!p_layer)
		CV_ERROR(CV_StsNullPtr, "Null double pointer");

	layer = *(CvCNNFullConnectLayer**) p_layer;

	if (!layer)
		return;
	if (!ICV_IS_CNN_FULLCONNECT_LAYER(layer))
		CV_ERROR(CV_StsBadArg, "Invalid layer");

	cvReleaseMat(&layer->exp2ssumWX);
	cvReleaseMat(&layer->weights);
	if (layer->hessian_diag)
		cvReleaseMat(&layer->hessian_diag);
	cvFree(p_layer);

	__CV_END__;
}

/****************************************************************************************/
static void icvReleaseCNNModel( void** ptr )
{
    CV_FUNCNAME("icvReleaseCNNModel");
    __CV_BEGIN__;

	if (!ptr)
		CV_ERROR(CV_StsNullPtr, "NULL double pointer");
	CV_ASSERT(CV_IS_CNN(*ptr));

	icvCNNModelRelease((CvCNNStatModel**) ptr);

	__CV_END__;
}
