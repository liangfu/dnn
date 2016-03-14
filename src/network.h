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

/** \class CvNetwork
 *  \brief CvNetwork class
 *  see member functions for detail
 */
class CvNetwork
{
public:
  CvNetwork(void);
  CvNetwork(int height, int width, int node, int cNode,
         double alpha, int maxiter, int batch_size);
  ~CvNetwork(void);

  /** \brief Create CNN models
   * Create CNN models.
   * @param a an integer argument.
   * @param s a constant character pointer.
   * @see publicVar()
   * @return The test results
   */
  void createNetwork( );

  /** \brief Save CNN parameters to a file
   * a normal member to save the CNN parameters to a file.
   * @param outFile the name of the output file.
   */
  void writeNetworkParams(string outFile);

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

  void readNetworkModel(string inFile){}
  
  void readNetworkSolver(string inFile){}

  /** \brief Load CNN parameters from a file
   * a normal member loading the parameters of CNN from a file.
   * @param inFile the file containing the CNN parameter info.
   */
  void readNetworkWeights(string inFile);

  /** \brief Train a CNN model
   * a normal member to train the CNN.
   * @param trainingData an integer argument.
   * @param responseMat a constant character pointer.
   */
  void trainNetwork(CvMat *trainingData, CvMat *responseMat);
};

#endif // __CV_CNNETWORK_H__

