/** \file CvNetwork.h
    \brief Called by file CovNN.cpp

    No Further Details.
*/

#ifndef __CV_CNNETWORK_H__
#define __CV_CNNETWORK_H__

#include "cv.h"
#include "highgui.h"
#include "ml.h"
// #include "_ml.h"
#include "cxcore.h"

using namespace std;

class CvCNNSolver
{
  char m_training_filename[1<<10];
  char m_response_filename[1<<10];
  char m_testing_filename[1<<10];
  char m_expected_filename[1<<10];
public:
  CvCNNSolver(char * solver_filename){
    CvFileStorage * fs = cvOpenFileStorage(solver_filename,0,CV_STORAGE_READ);
    CvFileNode * dnode = cvGetFileNodeByName(fs,0,"data");
    strcpy(m_training_filename,cvReadStringByName(fs,dnode,"training_filename"));
    strcpy(m_response_filename,cvReadStringByName(fs,dnode,"response_filename"));
    strcpy(m_testing_filename, cvReadStringByName(fs,dnode,"testing_filename"));
    strcpy(m_expected_filename,cvReadStringByName(fs,dnode,"expected_filename"));
    if (fs){cvReleaseFileStorage(&fs);fs=0;}
  }
  ~CvCNNSolver(){}
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

public:
  CvNetwork();
  // CvNetwork(int height, int width, int node, int cNode,
  //        double alpha, int maxiter, int batch_size);
  ~CvNetwork();

  CvCNNSolver * solver(){return m_solver;}

  /** \brief Create CNN models
   * Create CNN models.
   * @param a an integer argument.
   * @param s a constant character pointer.
   * @see publicVar()
   * @return The test results
   */
  void createNetwork( );

  /** \brief CNN models
   * a public variable.
   * Saved the CNN model in this variable.
   */
  CvCNNStatModel * m_cnn ;	

  // CvCNNStatModel * cnn_train;
  int            m_clipHeight, m_clipWidth;
  int            m_nNode, m_connectNode;
  int            m_max_iter;
  double         m_learningRate;
  int            m_batch_size;

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

