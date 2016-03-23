#include "network.h"

#include <stdio.h>

using namespace std;

CvNetwork::CvNetwork():m_solver(0),m_cnn(0)
{
}

CvNetwork::~CvNetwork(void)
{
  if ( m_solver ){ delete m_solver; m_solver=0; }
  if ( m_cnn ) { m_cnn->release( (CvCNNStatModel**)&m_cnn ); m_cnn=0; }
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
  int n_input_planes = 1;
  int input_height   = 1;
  int input_width    = 1;
  int n_output_planes = 1;
  int output_height   = 1;
  int output_width    = 1;
  
  m_cnn = (CvCNNStatModel*)cvCreateCNNStatModel(
    CV_STAT_MODEL_MAGIC_VAL|CV_CNN_MAGIC_VAL, sizeof(CvCNNStatModel));

  root = cvGetFileNodeByName(fs,root,"layers");
  for (int ii=0;;ii++){
    sprintf(nodename,"layer-%d",ii);
    node = cvGetFileNodeByName(fs,root,nodename);
    if (!node){break;}
    const char * predefined = cvReadStringByName(fs,node,"predefined","");
    const char * type = cvReadStringByName(fs,node,"type","");
    const char * name = cvReadStringByName(fs,node,"name","");
    const char * activation_desc = cvReadStringByName(fs,node,"activation_type","none");
    int activation_type = CV_CNN_HYPERBOLIC;
    if (!strcmp(activation_desc,"relu")){activation_type = CV_CNN_RELU;}
    else if (!strcmp(activation_desc,"tanh")){activation_type = CV_CNN_HYPERBOLIC;}
    else if (!strcmp(activation_desc,"logit")){activation_type = CV_CNN_LOGISTIC;}
    else if (!strcmp(activation_desc,"none")){activation_type = CV_CNN_NONE;}

    // parse layer-specific parameters
    if (strlen(predefined)>0){
      CvCNNLayer * predefined_layer = m_cnn->network->get_layer(m_cnn->network,predefined);
      if (ICV_IS_CNN_RECURRENT_LAYER(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvCNNRecurrentLayer * recurrent_layer = (CvCNNRecurrentLayer*)predefined_layer;
        layer = cvCreateCNNRecurrentLayer( predefined_layer->name, predefined_layer, 
          predefined_layer->n_input_planes, predefined_layer->n_output_planes, 
          recurrent_layer->n_hiddens, recurrent_layer->seq_length, time_index, lr_init, decay_type, 
          recurrent_layer->activation_type, NULL, NULL, NULL );
      }else if (ICV_IS_CNN_IMGCROPPING_LAYER(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvCNNImgCroppingLayer * crop_layer = (CvCNNImgCroppingLayer*)predefined_layer;
        CvCNNLayer * input_layer = crop_layer->input_layer;
        layer = cvCreateCNNImgCroppingLayer( name, input_layer, 
          crop_layer->n_output_planes, crop_layer->output_height, crop_layer->output_width, 
          time_index, lr_init, decay_type );
      }else{
        //layer = predefined_layer;
        assert(false); // unexpected!
      }
      n_input_planes = layer->n_output_planes; 
      input_height = layer->output_height; 
      input_width = layer->output_width;
    }else if (!strcmp(type,"Convolution")){ // convolution layer
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNConvolutionLayer( name, 
        n_input_planes, input_height, input_width, n_output_planes, ksize,
        lr_init, decay_type, NULL, NULL );
      n_input_planes = n_output_planes;
      input_height = input_height-ksize+1;
      input_width = input_width-ksize+1;
    }else if (!strcmp(type,"SubSampling")){ // max pooling layer
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNSubSamplingLayer( name, 
        n_input_planes, input_height, input_width, ksize,1,1,
        lr_init, decay_type, NULL);
      n_input_planes = n_output_planes;
      input_height = input_height/ksize;
      input_width = input_width/ksize;
    }else if (!strcmp(type,"FullConnect")){ // full connection layer
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      layer = cvCreateCNNFullConnectLayer( name, 
        n_input_planes, n_output_planes, 1, 1, 
        lr_init, decay_type, activation_type, NULL );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"ImgCropping")){ // image cropping layer
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)<1){
        LOGE("input layer name is required while defining ImgCropping layer."); exit(-1);
      }
      CvCNNLayer * input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      output_height   = cvReadIntByName(fs,node,"output_height",output_height);
      output_width    = cvReadIntByName(fs,node,"output_width",output_width);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateCNNImgCroppingLayer( name, input_layer, 
        n_output_planes, output_height, output_width, time_index, 
        lr_init, decay_type );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"Recurrent")){ // recurrent layer
      const int n_input_planes_default = n_input_planes * input_height * input_width;
      n_input_planes = cvReadIntByName(fs,node,"n_input_planes",n_input_planes_default);
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      const int n_hiddens_default = cvCeil(exp2((log2(n_input_planes)+log2(n_output_planes))*.5f));
      const int n_hiddens = cvReadIntByName(fs,node,"n_hiddens",n_hiddens_default);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateCNNRecurrentLayer( name, 0, 
        n_input_planes, n_output_planes, n_hiddens, seq_length, time_index, 
        lr_init, decay_type, activation_type, NULL, NULL, NULL );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"InputData")){ // data container layer
      n_input_planes = cvReadIntByName(fs,node,"n_input_planes",1);
      input_height   = cvReadIntByName(fs,node,"input_height",1);
      input_width    = cvReadIntByName(fs,node,"input_width",1);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      layer = cvCreateCNNInputDataLayer( name, 
        n_input_planes, input_height, input_width, seq_length,
        lr_init, decay_type );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else{
      fprintf(stderr,"ERROR: unknown layer type %s\n",type);
    }

    // add layer to network
    if (!m_cnn->network){m_cnn->network = cvCreateCNNetwork(layer); 
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
  
  layer=(CvCNNConvolutionLayer*)m_cnn->network->first_layer;
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
  const int n_layers = m_cnn->network->n_layers;
  CvCNNLayer * layer;
  CvFileStorage * fs = cvOpenFileStorage(outFile.c_str(),0,CV_STORAGE_WRITE);
  
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  
  layer=(CvCNNLayer*)m_cnn->network->first_layer;
  for (int ii=0;ii<n_layers;ii++,layer=layer->next_layer){
    if (ICV_IS_CNN_RECURRENT_LAYER(layer)){
      CvCNNRecurrentLayer * rnnlayer = (CvCNNRecurrentLayer*)layer;
      cvWrite(fs,(string(rnnlayer->name)+string("_Wxh")).c_str(),rnnlayer->Wxh);
      cvWrite(fs,(string(rnnlayer->name)+string("_Whh")).c_str(),rnnlayer->Whh);
      cvWrite(fs,(string(rnnlayer->name)+string("_Why")).c_str(),rnnlayer->Why);
    }
  }
  // cvWrite(fs,"conv1",layer->weights);
  // cvWrite(fs,"conv2",layer->next_layer->next_layer->weights);
  // cvWrite(fs,"softmax1",layer->next_layer->next_layer->next_layer->next_layer->weights);
  // cvWrite(fs,"softmax2",layer->next_layer->next_layer->next_layer->next_layer->next_layer->weights);

  cvReleaseFileStorage(&fs);
}

void CvNetwork::train(CvMat *trainingData, CvMat *responseMat)
{
  int i, j;	
  CvCNNStatModelParams params;
  assert(CV_MAT_TYPE(trainingData->type)==CV_32F);

  CvCNNLayer * last_layer = cvGetCNNLastLayer(m_cnn->network);
  int n_outputs = last_layer->n_output_planes;
  if (ICV_IS_CNN_RECURRENT_LAYER(last_layer)){
    n_outputs *= ((CvCNNRecurrentLayer*)last_layer)->seq_length;
  }

  params.cls_labels = cvCreateMat( n_outputs, n_outputs, CV_32FC1 );
  cvSet(params.cls_labels,cvScalar(1));
  params.etalons = cvCreateMat( n_outputs, n_outputs, CV_32FC1 );
  cvSetIdentity(params.etalons,cvScalar(2));
  cvSubS(params.etalons,cvScalar(1),params.etalons);

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
  return top1;
}



