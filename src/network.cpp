#include "network.h"

#include <stdio.h>

using namespace std;

void cvSaveCategorialResult(CvDNNLayer * last_layer, CvMat * input, const char * output_filename);

Network::Network():m_solver(0),m_cnn(0)
{
}

Network::~Network(void)
{
  if ( m_solver ){ delete m_solver; m_solver=0; }
  if ( m_cnn ) { m_cnn->release( (CvDNNStatModel**)&m_cnn ); m_cnn=0; }
}

void Network::loadModel(string inFile)
{
  CV_FUNCNAME("Network::loadModel");
  __BEGIN__;
  CvDNNLayer * layer = 0;
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  CvFileNode * root = cvGetRootFileNode( fs );
  CvFileNode * node = 0;
  float lr_init = m_solver->lr_init();
  int decay_type = m_solver->decay_type();
  float momentum_ratio = m_solver->momentum_ratio();

  node = cvGetFileNodeByName(fs,root,"data");
  int n_input_planes = 1;
  int input_height   = 1;
  int input_width    = 1;
  int n_output_planes = 1;
  int output_height   = 1;
  int output_width    = 1;
  
  int dtype = CV_32F;
  const char * dtypename = cvReadStringByName(fs,node,"dtype","float");
  if (!strcmp(dtypename,"float")){dtype=CV_32F;
  }else if (!strcmp(dtypename,"double")){dtype=CV_64F;
  }else{fprintf(stderr,"Error: unknown dtype name `%s`\n",dtypename);exit(-1);}
  
  m_cnn = (CvDNNStatModel*)cvCreateStatModel(
    CV_STAT_MODEL_MAGIC_VAL|CV_DNN_MAGIC_VAL, sizeof(CvDNNStatModel));

  CvFileNode * layers = cvGetFileNodeByName(fs,0,"layers");
  assert(CV_NODE_IS_SEQ(layers->tag));
  CvSeq * seq = layers->data.seq;
  int ii, total = seq->total;
  CvSeqReader reader;
  cvStartReadSeq( seq, &reader, 0 );
  
  for (ii=0;ii<total;ii++){
    node = (CvFileNode*)reader.ptr;
    if (!node){break;}
    const char * predefined = cvReadStringByName(fs,node,"predefined","");
    const char * type = cvReadStringByName(fs,node,"type","");
    const char * name = cvReadStringByName(fs,node,"name","");
    const int visualize = cvReadIntByName(fs,node,"visualize",0);
    const char * activation = cvReadStringByName(fs,node,"activation","none");

    // parse layer-specific parameters
    if (strlen(predefined)>0){
      CvDNNLayer * predefined_layer = m_cnn->network->get_layer(m_cnn->network,predefined);
      if (icvIsSimpleRNNLayer(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvDNNSimpleRNNLayer * recurrent_layer = (CvDNNSimpleRNNLayer*)predefined_layer;
        layer = cvCreateSimpleRNNLayer( 
          predefined_layer->dtype, predefined_layer->name, predefined_layer, 
          predefined_layer->n_input_planes, predefined_layer->n_output_planes, 
          recurrent_layer->n_hiddens, recurrent_layer->seq_length, time_index, lr_init, decay_type, 
          recurrent_layer->activation, NULL, NULL, NULL );
        if (((CvDNNSimpleRNNLayer*)layer)->Wxh){
          cvReleaseMat(&((CvDNNSimpleRNNLayer*)layer)->Wxh);((CvDNNSimpleRNNLayer*)layer)->Wxh=0;}
        if (((CvDNNSimpleRNNLayer*)layer)->Whh){
          cvReleaseMat(&((CvDNNSimpleRNNLayer*)layer)->Whh);((CvDNNSimpleRNNLayer*)layer)->Whh=0;}
        if (((CvDNNSimpleRNNLayer*)layer)->Why){
          cvReleaseMat(&((CvDNNSimpleRNNLayer*)layer)->Why);((CvDNNSimpleRNNLayer*)layer)->Why=0;}
      }else if (icvIsSpatialTransformLayer(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvDNNSpatialTransformLayer * this_layer = (CvDNNSpatialTransformLayer*)predefined_layer;
        CvDNNLayer * input_layer = (this_layer->input_layers.size()>0?this_layer->input_layers[0]:0);
        layer = cvCreateSpatialTransformLayer( 
          this_layer->dtype, this_layer->name, this_layer->visualize, input_layer, 
          this_layer->n_output_planes, this_layer->output_height, this_layer->output_width, 
          this_layer->seq_length, time_index, this_layer->init_learn_rate, this_layer->decay_type );
      }else if (icvIsDenseLayer(predefined_layer)){
        CvDNNDenseLayer * this_layer = (CvDNNDenseLayer*)predefined_layer;
        layer = cvCreateDenseLayer( 
          this_layer->dtype, this_layer->name, this_layer->visualize, 
          this_layer->input_layers.size()>0?this_layer->input_layers[0]:0, 
          this_layer->n_input_planes, this_layer->n_output_planes, 
          this_layer->init_learn_rate, this_layer->decay_type, this_layer->activation, NULL );
      }else if (icvIsConvolutionLayer(predefined_layer)){
        CvDNNConvolutionLayer * this_layer = (CvDNNConvolutionLayer*)predefined_layer;
        layer = cvCreateConvolutionLayer( 
          this_layer->dtype, this_layer->name, predefined_layer, this_layer->visualize, 
          this_layer->input_layers.size()>0?this_layer->input_layers[0]:0, 
          this_layer->n_input_planes, this_layer->input_height, this_layer->input_width,
          this_layer->n_output_planes, this_layer->K,
          this_layer->init_learn_rate, this_layer->decay_type, this_layer->activation, NULL, NULL );
      }else if (icvIsMaxPoolingLayer(predefined_layer)){
        CvDNNMaxPoolingLayer * this_layer = (CvDNNMaxPoolingLayer*)predefined_layer;
        layer = cvCreateMaxPoolingLayer( 
          this_layer->dtype, this_layer->name, this_layer->visualize,
          this_layer->n_input_planes, this_layer->input_height, this_layer->input_width,
          this_layer->sub_samp_scale, this_layer->init_learn_rate, this_layer->decay_type, 0 );
      }else if (icvIsTimeDistributedLayer(predefined_layer)){
        int time_index = cvReadIntByName(fs,node,"time_index",0);
        CvDNNTimeDistributedLayer * this_layer = (CvDNNTimeDistributedLayer*)predefined_layer;
        CvDNNLayer * input_layer = (this_layer->input_layers.size()>0?this_layer->input_layers[0]:0);
        layer = cvCreateTimeDistributedLayer(
          this_layer->dtype, this_layer->name, this_layer->visualize, input_layer, 
          this_layer->n_output_planes, this_layer->output_height, this_layer->output_width, 
          this_layer->seq_length, time_index, this_layer->init_learn_rate, this_layer->decay_type );
      }else{
        assert(false);
      }
      // release weights matrix, use the one on predefined layer instead.
      if (layer->weights){cvReleaseMat(&layer->weights);layer->weights=0;}
      n_input_planes = layer->n_output_planes; 
      input_height = layer->output_height; 
      input_width = layer->output_width;
    }else if (!strcmp(type,"Convolution")){ // convolution layer
      CvDNNLayer * input_layer = 0; 
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)>0){
        input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      }
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateConvolutionLayer( dtype, name, 0, visualize, input_layer, 
        n_input_planes, input_height, input_width, n_output_planes, ksize,
        lr_init, decay_type, activation, NULL, NULL );
      if (input_layer){input_layer->output_layers.push_back(layer);}
      n_input_planes = n_output_planes;
      input_height = input_height-ksize+1;
      input_width = input_width-ksize+1;
    }else if (!strcmp(type,"MaxPooling")){ // max pooling layer
      CvDNNLayer * input_layer = 0; 
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)>0){
        input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      }
      int ksize = cvReadIntByName(fs,node,"ksize");
      layer = cvCreateMaxPoolingLayer( dtype, name, visualize,
        n_input_planes, input_height, input_width, ksize,
        lr_init, decay_type, NULL);
      if (input_layer){input_layer->output_layers.push_back(layer);}
      n_input_planes = n_output_planes;
      input_height = input_height/ksize;
      input_width = input_width/ksize;
    }else if (!strcmp(type,"Dense")){ // full connection layer
      CvDNNLayer * input_layer = 0; 
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)>0){
        input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
        if (!input_layer){LOGE("input_layer [%s] not found.",input_layer_name);exit(-1);}
        n_input_planes = input_layer->n_output_planes*input_layer->output_height*input_layer->output_width;
      }else{
        n_input_planes = n_input_planes*input_height*input_width;
      }
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes");
      layer = cvCreateDenseLayer( dtype, name, visualize, input_layer, 
        n_input_planes, n_output_planes, 
        lr_init, decay_type, activation, NULL );
      if (input_layer){input_layer->output_layers.push_back(layer);}
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"SpatialTransform")){ // image cropping layer
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)<1){
        LOGE("input layer name is required while defining SpatialTransform layer."); exit(-1);}
      CvDNNLayer * input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      if (!input_layer){
        LOGE("input layer is not found while defining SpatialTransform layer."); exit(-1);}
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      output_height = cvReadIntByName(fs,node,"output_height",output_height);
      output_width = cvReadIntByName(fs,node,"output_width",output_width);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateSpatialTransformLayer( dtype, name, visualize, input_layer, 
        n_output_planes, output_height, output_width, seq_length, time_index, 
        lr_init, decay_type );
      n_input_planes = n_output_planes;
      input_height = output_height; input_width = output_width;
    }else if (!strcmp(type,"TimeDistributed")){ // temporal slice layer
      const char * input_layer_name = cvReadStringByName(fs,node,"input_layer","");
      if (strlen(input_layer_name)<1){
        LOGE("input layer name is required while defining TimeDistributed layer."); exit(-1);}
      CvDNNLayer * input_layer = m_cnn->network->get_layer(m_cnn->network,input_layer_name);
      if (!input_layer){
        LOGE("input layer is not found while defining TimeDistributed layer."); exit(-1);}
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      output_height = cvReadIntByName(fs,node,"output_height",output_height);
      output_width = cvReadIntByName(fs,node,"output_width",output_width);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateTimeDistributedLayer( dtype, name, visualize, input_layer, 
        n_output_planes, output_height, output_width, seq_length, time_index, 
        lr_init, decay_type );
      n_input_planes = n_output_planes;
      input_height = output_height; input_width = output_width;
    }else if (!strcmp(type,"SimpleRNN")){ // recurrent layer
      const int n_input_planes_default = n_input_planes * input_height * input_width;
      n_input_planes = cvReadIntByName(fs,node,"n_input_planes",n_input_planes_default);
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      const int n_hiddens_default = cvCeil(exp2((log2(n_input_planes)+log2(n_output_planes))*.5f));
      const int n_hiddens = cvReadIntByName(fs,node,"n_hiddens",n_hiddens_default);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateSimpleRNNLayer( dtype, name, 0, 
        n_input_planes, n_output_planes, n_hiddens, seq_length, time_index, 
        lr_init, decay_type, activation, NULL, NULL, NULL );
      n_input_planes = n_output_planes; input_height = 1; input_width = 1;
    }else if (!strcmp(type,"Input")){ // data container layer
      n_input_planes = cvReadIntByName(fs,node,"n_input_planes",n_input_planes);
      input_height   = cvReadIntByName(fs,node,"input_height",input_height);
      input_width    = cvReadIntByName(fs,node,"input_width",input_width);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      layer = cvCreateInputLayer( dtype, name, 
        n_input_planes, input_height, input_width, seq_length,
        lr_init, decay_type );
    }else if (!strcmp(type,"RepeatVector")){ // data container layer
      n_input_planes = cvReadIntByName(fs,node,"n_input_planes",n_input_planes);
      input_height   = cvReadIntByName(fs,node,"input_height",input_height);
      input_width    = cvReadIntByName(fs,node,"input_width",input_width);
      const int seq_length = cvReadIntByName(fs,node,"seq_length",1);
      const int time_index = cvReadIntByName(fs,node,"time_index",0);
      layer = cvCreateRepeatVectorLayer( dtype, name, 
        n_input_planes, input_height, input_width, seq_length, time_index, 
        lr_init, decay_type );
    }else if (!strcmp(type,"Merge")){ // merge multiple data source  layer
      int n_input_layers = 1;
      const char * input_layer_names = cvReadStringByName(fs,node,"input_layers","");
      n_output_planes = cvReadIntByName(fs,node,"n_output_planes",1);
      static const int max_input_layers = 100;
      CvDNNLayer ** input_layers = new CvDNNLayer*[max_input_layers];
      char * input_layer_name = strtok((char*)input_layer_names," ,");
      input_layers[0] = m_cnn->network->get_layer(m_cnn->network, input_layer_name);
      int output_plane_count = input_layers[0]->n_output_planes;
      for (int ii=1; ii<max_input_layers; ii++){
        input_layer_name = strtok(0," ,");
        if (!input_layer_name){break;}else{n_input_layers++;}
        input_layers[ii] = m_cnn->network->get_layer(m_cnn->network, input_layer_name);
        output_plane_count += input_layers[ii]->n_output_planes;
      }
      if (output_plane_count!=n_output_planes){
        CV_ERROR(CV_StsBadArg,"Invalid definition of `input_layers` in Merge layer, "
                 "`n_output_planes` should equal to sum of output_planes of input_layers.");
      }
      layer = cvCreateMergeLayer( dtype, name, visualize, 
        n_input_layers, input_layers, n_output_planes, lr_init, decay_type );
      for (int lidx=0;lidx<n_input_layers;lidx++){input_layers[lidx]->output_layers.push_back(layer);}
      if (input_layers){delete [] input_layers; input_layers = 0;}
      n_input_planes = n_output_planes;
    }else{
      fprintf(stderr,"ERROR: unknown layer type %s\n",type);
    }

    // add layer to network
    if (!m_cnn->network){m_cnn->network = cvCreateNetwork(layer); 
    }else{m_cnn->network->add_layer( m_cnn->network, layer );}

    CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
  }

  if (fs){cvReleaseFileStorage(&fs);fs=0;}
  __END__;
}

void Network::loadWeights(string inFile)
{
  if (m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  CvFileStorage * fs = cvOpenFileStorage(inFile.c_str(),0,CV_STORAGE_READ);
  m_cnn->network->read(m_cnn->network,fs);
  cvReleaseFileStorage(&fs);
}

void Network::saveWeights(string outFile)
{
  if(m_cnn == NULL){fprintf(stderr,"ERROR: CNN has not been built yet\n");exit(0);}
  CvFileStorage * fs = cvOpenFileStorage(outFile.c_str(),0,CV_STORAGE_WRITE);
  m_cnn->network->write(m_cnn->network,fs);
  cvReleaseFileStorage(&fs);
}

void Network::train(CvMat *trainingData, CvMat *responseMat)
{
  int i, j;	
  CvDNNStatModelParams params;
  assert(CV_MAT_TYPE(trainingData->type)==CV_32F);

  CvDNNLayer * last_layer = cvGetCNNLastLayer(m_cnn->network);
  int n_outputs = last_layer->n_output_planes;
  if (icvIsSimpleRNNLayer(last_layer)){
    n_outputs *= ((CvDNNSimpleRNNLayer*)last_layer)->seq_length;
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
  params.grad_estim_type=CV_DNN_GRAD_ESTIM_RANDOM;
  params.nepochs = m_solver->nepochs();
  params.validate_ratio = m_solver->validate_ratio();
  params.momentum_ratio = m_solver->momentum_ratio();

  if (CV_MAT_TYPE(responseMat->type)!=CV_32F){
    CvMat * tmp = cvCreateMat(responseMat->rows,responseMat->cols,CV_32F);
    cvConvert(responseMat,tmp);
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,tmp,&params,0,0,0,0);
    cvReleaseMat(&tmp);
  }else{
    m_cnn = cvTrainCNNClassifier( trainingData, CV_ROW_SAMPLE,responseMat,&params,0,0,0,0);
  }
}

float Network::evaluate(CvMat * testing, CvMat * expected, int nsamples, const char * predicted_filename)
{
  CV_FUNCNAME("Network::evaluate");
  float top1=0;
  __BEGIN__;
  CvMat samples = cvMat(nsamples, testing->cols, CV_32F, testing->data.fl+nsamples*testing->cols);
  CvDNNLayer * last_layer = m_cnn->network->get_last_layer(m_cnn->network);
  CvMat * result = cvCreateMat(nsamples*last_layer->seq_length, last_layer->n_output_planes, CV_32F);
  if (expected){
    CV_ASSERT(expected->cols==last_layer->n_output_planes*last_layer->seq_length);
  }

  // testing data
  cvGetRows(testing,&samples,0,nsamples);
  m_cnn->predict(m_cnn,&samples,result,m_solver->batch_size());

  // compute loss & accuracy, print progress
  CvMat * expected_submat = cvCreateMat(nsamples*last_layer->seq_length,last_layer->n_output_planes,CV_32F);
  CvMat expected_submat_hdr,expected_submat_reshape_hdr;
  if (expected){
    CV_ASSERT(expected->cols==last_layer->n_output_planes*last_layer->seq_length);
    cvGetRows(expected,&expected_submat_hdr,0,nsamples);
    cvReshape(&expected_submat_hdr,&expected_submat_reshape_hdr,0,nsamples*last_layer->seq_length);
    cvCopy(&expected_submat_reshape_hdr,expected_submat);
    float trloss = cvNorm(result, expected_submat)/float(nsamples);
    top1 = m_cnn->network->eval(last_layer, result, expected_submat);
    static double sumloss = trloss;
    static double sumacc  = top1;
    fprintf(stderr, "sumacc: %.1f%%[%.1f%%], sumloss: %f\n", sumacc,top1,sumloss);
  }
  {
    List<int> output_planes; int output_planes_count=0;
    if (icvIsMergeLayer(last_layer)){
      for (int ii=0;ii<last_layer->input_layers.size();ii++){
        output_planes.push_back(output_planes_count+last_layer->input_layers[ii]->n_output_planes);
        output_planes_count+=last_layer->input_layers[ii]->n_output_planes;
      }
    }
    if (nsamples<=5){fprintf(stderr,"output:\n");
      CvMat result_submat_hdr;
      for (int ii=0;ii<result->rows;ii++){
        cvGetSubRect(result,&result_submat_hdr,cvRect(0,ii,output_planes[0],1));
        cvPrintf(stderr,"%.1f ", &result_submat_hdr);
        for (int jj=1;jj<output_planes.size();jj++){
          cvGetSubRect(result,&result_submat_hdr,
                       cvRect(output_planes[jj-1],ii,output_planes[jj]-output_planes[jj-1],1));
          cvPrintf(stderr,"%.1f ", &result_submat_hdr);
        }
      }
    }
    if (strlen(predicted_filename)>0){
      cvSave(predicted_filename,result);
      char output_filename[1024]={0,}; 
      cvGetBaseName((char*)predicted_filename,(char*)output_filename); strcat(output_filename,".txt");
      cvSaveCategorialResult(last_layer, result, (char*)output_filename);
      LOGI("prediction result saved to: %s.", predicted_filename);
    }
    if (expected && nsamples<=5){
      fprintf(stderr,"expected:\n");cvPrintf(stderr,"%.1f ", &expected_submat_reshape_hdr);
    }
  }
  cvReleaseMat(&expected_submat);
  
  cvReleaseMat(&result);
  __END__;
  return top1;
}

void cvSaveCategorialResult(CvDNNLayer * last_layer, CvMat * input, const char * output_filename)
{
  List<int> output_planes; int output_planes_count=0;
  double minval,maxval; CvPoint minloc,maxloc;
  if (icvIsMergeLayer(last_layer)){
    for (int ii=0;ii<last_layer->input_layers.size();ii++){
      output_planes.push_back(output_planes_count+last_layer->input_layers[ii]->n_output_planes);
      output_planes_count+=last_layer->input_layers[ii]->n_output_planes;
    }
  }

  FILE * fp = fopen(output_filename,"wt");
  CvMat input_submat_hdr;
  for (int ii=0;ii<input->rows;ii++){
    cvGetSubRect(input,&input_submat_hdr,cvRect(0,ii,output_planes[0],1));
    cvMinMaxLoc(&input_submat_hdr,0,&maxval,0,&maxloc);
    CvMat * maxvals = cvCreateMat(1,maxloc.x+2,CV_64F);
    CvMat * maxlocs = cvCreateMat(1,maxloc.x+2,CV_32S);
    for (int jj=1;jj<maxlocs->cols+1;jj++){
      cvGetSubRect(input,&input_submat_hdr,
                   cvRect(output_planes[jj-1],ii,output_planes[jj]-output_planes[jj-1],1));
      cvMinMaxLoc(&input_submat_hdr,0,&maxvals->data.db[jj-1],0,&maxloc);
      maxlocs->data.i[jj-1]=maxloc.x;
    }
    cvMinMaxLoc(maxvals,&minval,0,0,0);
    float avgval=cvAvg(maxvals).val[0];
    if (minval>.5f){cvPrintf(fp,"%d",maxlocs);}else{fprintf(fp,"--\n");} 
    cvReleaseMat(&maxvals);
    cvReleaseMat(&maxlocs);
  }
  fclose(fp);
}

