/** \file CvNetwork.h
    \brief Called by file CovNN.cpp

    No Further Details.
*/

#ifndef __CV_CNNETWORK_H__
#define __CV_CNNETWORK_H__

#include "cv.h"
#include "highgui.h"
#include "dnn.h"
// #include "_ml.h"
#include "cxcore.h"
#include "cvext_c.h"

#include <stdio.h>
#include <stdlib.h>

using namespace std;

class CvCNNSolver
{
  float m_lr_init;
  int m_decay_type;// = CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;
  int m_maxiter;
  int m_batch_size;
  float m_validate_ratio;

  char m_model_filename[1<<10];
  char m_weights_filename[1<<10];

  char m_training_filename[1<<10];
  char m_response_filename[1<<10];
  char m_testing_filename[1<<10];
  char m_expected_filename[1<<10];
public:
  CvCNNSolver(char * solver_filename):
    m_lr_init(.0001f),m_decay_type(CV_CNN_LEARN_RATE_DECREASE_SQRT_INV),
    m_maxiter(1),m_batch_size(1),m_validate_ratio(0.1f)
  {
    CvFileStorage * fs = cvOpenFileStorage(solver_filename,0,CV_STORAGE_READ);
    if (!fs){fprintf(stderr,"error: solver file %s not exist!\n",solver_filename); exit(-1);}
    CvFileNode * node = cvGetFileNodeByName(fs,0,"data");
    strcpy(m_training_filename,cvReadStringByName(fs,node,"training_filename"));
    strcpy(m_response_filename,cvReadStringByName(fs,node,"response_filename"));
    strcpy(m_testing_filename, cvReadStringByName(fs,node,"testing_filename"));
    strcpy(m_expected_filename,cvReadStringByName(fs,node,"expected_filename"));
    node = cvGetFileNodeByName(fs,0,"network");
    strcpy(m_model_filename,cvReadStringByName(fs,node,"model_filename"));
    strcpy(m_weights_filename,cvReadStringByName(fs,node,"weights_filename"));
    m_lr_init = cvReadRealByName(fs,node,"lr_init");
    // const char * decay_desc = cvReadStringByName(fs,node,"decay_type");
    // if (!strcmp(decay_desc,"invsqrt")){m_decay_type=CV_CNN_LEARN_RATE_DECREASE_SQRT_INV;}
    m_maxiter = cvReadIntByName(fs,node,"maxiter");
    m_batch_size = cvReadIntByName(fs,node,"batch_size");
    m_validate_ratio = cvReadRealByName(fs,node,"validate_ratio");
    if (fs){cvReleaseFileStorage(&fs);fs=0;}
  }
  ~CvCNNSolver(){}

  float lr_init(){return m_lr_init;}
  int decay_type(){return m_decay_type;}
  int maxiter(){return m_maxiter;}
  int batch_size(){return m_batch_size;}
  float validate_ratio(){return m_validate_ratio;}

  char * model_filename(){return (char*)m_model_filename;}
  char * weights_filename(){return (char*)m_weights_filename;}
  
  char * training_filename(){return (char*)m_training_filename;}
  char * response_filename(){return (char*)m_response_filename;}
  char * testing_filename (){return (char*)m_testing_filename;}
  char * expected_filename(){return (char*)m_expected_filename;}
};

/** \class CvNetwork
 *  \brief CvNetwork class
 *  see member functions for detail
 */
class CvNetwork
{
  CvCNNSolver * m_solver;
  CvCNNStatModel * m_cnn ;	

public:
  CvNetwork();
  // CvNetwork(int height, int width, int node, int cNode,
  //        double alpha, int maxiter, int batch_size);
  ~CvNetwork();

  CvCNNSolver * solver(){return m_solver;}

  /** \brief CNN models
   * a public variable.
   * Saved the CNN model in this variable.
   */
  // int            m_clipHeight, m_clipWidth;
  // int            m_nNode, m_connectNode;
  // int            m_max_iter;
  // double         m_learningRate;
  // int            m_batch_size;

  void loadModel(string inFile);
  
  void loadSolver(string inFile){
    if (m_solver){delete m_solver;m_solver=0;}
    m_solver = new CvCNNSolver((char*)inFile.c_str());
  }

  /** \brief Load CNN parameters from a file
   * a normal member loading the parameters of CNN from a file.
   * @param inFile the file containing the CNN parameter info.
   */
  void loadWeights(string inFile);

  /** \brief Save CNN parameters to a file
   * a normal member to save the CNN parameters to a file.
   * @param outFile the name of the output file.
   */
  void saveWeights(string outFile);

  /** \brief Train a CNN model
   * a normal member to train the CNN.
   * @param trainingData an integer argument.
   * @param responseMat a constant character pointer.
   */
  void train(CvMat *trainingData, CvMat *responseMat);

  float evaluate(CvMat * testing, CvMat * expected, int nsamples);
};

#endif // __CV_CNNETWORK_H__

