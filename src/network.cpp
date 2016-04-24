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

  CvFileNode * layers = cvGetFileNodeByName(fs,0,"layers");
  assert(CV_NODE_IS_SEQ(layers->tag));
  CvSeq * seq = layers->data.seq;
  int ii, total = seq->total;
  CvSeqReader reader;
  cvStartReadSeq( seq, &reader, 0 );
  
  CvStringHashNode * layer_key = cvGetHashedKey( fs, "name", -1, 1 );
  for (ii=0;ii<total;ii++){
    node = (CvFileNode*)reader.ptr;
    if (!node){break;}
    const char * predefined = cvReadStringByName(fs,node,"predefined","");
    const char * type = cvReadStringByName(fs,node,"type","");
    const char * name = cvReadStringByName(fs,node,"name","");
    const int visualize = cvReadIntByName(fs,node,"visualize",0);
    const char * activation_type = cvReadStringByName(fs,node,"activation_type","none");

    // parse layer-specific parameters
    if (strlen(predefined)>0){
      CvCNNLayer * predefined_layer = m_cnn->network->get_layer(m_cnn->network,predefined);
      if (ICV_IS_CNN_RECURRENTNN_LAYER(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvCNNRecurrentLayer * recurrent_layer = (CvCNNRecurrentLayer*)predefined_layer;
        layer = cvCreateCNNRecurrentLayer( predefined_layer->name, predefined_layer, 
          predefined_layer->n_input_planes, predefined_layer->n_output_planes, 
          recurrent_layer->n_hiddens, recurrent_layer->seq_length, time_index, lr_init, decay_type, 
          recurrent_layer->activation_type, NULL, NULL, NULL );
      }else if (ICV_IS_CNN_IMGCROPPING_LAYER(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvCNNImgCroppingLayer * this_layer = (CvCNNImgCroppingLayer*)predefined_layer;
        CvCNNLayer * input_layer = this_layer->input_layer;
        layer = cvCreateCNNImgCroppingLayer( this_layer->name, this_layer->visualize, input_layer, 
          this_layer->n_output_planes, this_layer->output_height, this_layer->output_width, 
          time_index, this_layer->init_learn_rate, this_layer->decay_type );
      }else if (ICV_IS_CNN_FULLCONNECT_LAYER(predefined_layer)){
        CvCNNFullConnectLayer * this_layer = (CvCNNFullConnectLayer*)predefined_layer;
        layer = cvCreateCNNFullConnectLayer( this_layer->name, this_layer->visualize, 
          this_layer->input_layer, this_layer->n_input_planes, this_layer->n_output_planes, 
          this_layer->init_learn_rate, this_layer->decay_type, this_layer->activation_type, NULL );
      }else if (ICV_IS_CNN_CONVOLUTION_LAYER(predefined_layer)){
        CvCNNConvolutionLayer * this_layer = (CvCNNConvolutionLayer*)predefined_layer;
        layer = cvCreateCNNConvolutionLayer( this_layer->name, this_layer->visualize,
          this_layer->n_input_planes, this_layer->input_height, this_layer->input_width,
          this_layer->n_output_planes, this_layer->K,
          this_layer->init_learn_rate, this_layer->decay_type, NULL, NULL );
      }else if (ICV_IS_CNN_SUBSAMPLING_LAYER(predefined_layer)){
        CvCNNSubSamplingLayer * this_layer = (CvCNNSubSamplingLayer*)predefined_layer;
        layer = cvCreateCNNSubSamplingLayer( this_layer->name, this_layer->visualize,
          this_layer->n_input_planes, this_layer->input_height, this_layer->input_width,
          this_layer->sub_samp_scale, this_layer->init_learn_rate, this_layer->decay_type, 0 );
      }else{
        assert(false);
      }
      n_input_planes = layer->n_output_planes; 
      input_height = layer->output_height; 
      input_width = layer->output_width;
    }else if (!strcmp(type,"Convolution")){ // convolution layer
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNConvolutionLayer( name, visualize,
        n_input_planes, input_height, input_width, n_output_planes, ksize,
        lr_init, decay_type, NULL, NULL );
      n_input_planes = n_output_planes;
      input_height = input_height-ksize+1;
      input_width = input_width-ksize+1;
    }else if (!strcmp(type,"SubSampling")){ // max pooling layer
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateCNNSubSamplingLayer( name, visualize,
        n_input_planes, input_height, input_width, ksize,
        lr_init, decay_type, NULL);
      n_input_planes = n_output_planes;
      input_height = input_height/ksize;
      input_width = input_width/ksize;
    }else if (!strcmp(type,"FullConnect")){ // full connection layer
      CvCNNLayer * input_layer = 0; 
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)>0){
        input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
        n_input_planes = input_layer->n_output_planes;
      }
      n_input_planes = n_input_planes*input_height*input_width;
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      layer = cvCreateCNNFullConnectLayer( name, visualize, input_layer, 
        n_input_planes, n_output_planes, 
        lr_init, decay_type, activation_type, NULL );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"ImgCropping")){ // image cropping layer
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)<1){
        LOGE("input layer name is required while defining ImgCropping layer."); exit(-1);}
      CvCNNLayer * input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      if (!input_layer){
        LOGE("input layer is not found while defining ImgCropping layer."); exit(-1);}
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      output_height = cvReadIntByName(fs,node,"output_height",output_height);
      output_width = cvReadIntByName(fs,node,"output_width",output_width);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateCNNImgCroppingLayer( name, visualize, input_layer, 
        n_output_planes, output_height, output_width, time_index, 
        lr_init, decay_type );
      n_input_planes = n_output_planes; // input_height = 1; input_width = 1;
      input_height = output_height; input_width = output_width;
    }else if (!strcmp(type,"RecurrentNN")){ // recurrent layer
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
    }else if (!strcmp(type,"MultiTarget")){ // merge multiple data source  layer
      int n_input_layers = cvReadIntByName(fs,node,"n_input_layers",1);
      const char * input_layer_names = cvReadStringByName(fs,node,"input_layers","");
      if (!(n_input_layers>=1 && n_input_layers<=100)){
        LOGE("Invalid number of input layers [%d] while defining MultiTarget layer.",
             n_input_layers);exit(-1);}
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      CvCNNLayer ** input_layers = new CvCNNLayer*[n_input_layers];
      char * input_layer_name = strtok((char*)input_layer_names," ,");
      input_layers[0] = m_cnn->network->get_layer(m_cnn->network, input_layer_name);
      for (int ii=1; ii<n_input_layers; ii++){
        input_layer_name = strtok(0," ,");
        input_layers[ii] = m_cnn->network->get_layer(m_cnn->network, input_layer_name);
      }
      layer = cvCreateCNNMultiTargetLayer( name, 
        n_input_layers, input_layers, n_output_planes, lr_init, decay_type );
      if (input_layers){delete [] input_layers; input_layers = 0;}
    }else{
      fprintf(stderr,"ERROR: unknown layer type %s\n",type);
    }

    // add layer to network
    if (!m_cnn->network){m_cnn->network = cvCreateCNNetwork(layer); 
    }else{m_cnn->network->add_layer( m_cnn->network, layer );}

    CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
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
    if (ICV_IS_CNN_RECURRENTNN_LAYER(layer)){
      CvCNNRecurrentLayer * rnnlayer = (CvCNNRecurrentLayer*)layer;
      char xhstr[1024],hhstr[1024],hystr[1024];
      sprintf(xhstr,"%s_step%d_Wxh",rnnlayer->name,rnnlayer->time_index);
      sprintf(hhstr,"%s_step%d_Whh",rnnlayer->name,rnnlayer->time_index);
      sprintf(hystr,"%s_step%d_Why",rnnlayer->name,rnnlayer->time_index);
      if (rnnlayer->Wxh){
        cvWrite(fs,xhstr,rnnlayer->Wxh);
        cvWrite(fs,hhstr,rnnlayer->Whh);
        cvWrite(fs,hystr,rnnlayer->Why);
      }else{assert(!rnnlayer->Wxh && !rnnlayer->Whh && !rnnlayer->Why);}
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
  if (ICV_IS_CNN_RECURRENTNN_LAYER(last_layer)){
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
  CvMat expected_submat_hdr;
  CvMat * expected_sorted = cvCreateMat(nsamples,expected->cols,CV_32F);
  CvMat * expected_index  = cvCreateMat(nsamples,expected->cols,CV_32S);
  CvMat expected_index_submat_hdr;
  CvMat * expected_transposed = cvCreateMat(1,result->cols,CV_32S);
  CvMat * indtop1res = cvCreateMat(1,result->cols,CV_8U);
  // testing data
  cvGetRows(testing,samples,0,nsamples);
  m_cnn->predict(m_cnn,samples,result);
  cvSort(result,sorted,indices,CV_SORT_DESCENDING|CV_SORT_EVERY_COLUMN);
  cvGetRow(indices,indtop1,0);
  // expected data
  cvGetRows(expected,&expected_submat_hdr,0,nsamples);
  cvSort(&expected_submat_hdr,expected_sorted,expected_index,CV_SORT_DESCENDING|CV_SORT_EVERY_ROW);
  cvGetCol(expected_index,&expected_index_submat_hdr,0);
  cvTranspose(&expected_index_submat_hdr,expected_transposed);
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
  cvReleaseMat(&expected_sorted);
  cvReleaseMat(&expected_index);
  cvReleaseMat(&expected_transposed);
  cvReleaseMat(&indtop1res);
  return top1;
}



