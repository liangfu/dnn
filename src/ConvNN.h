/** \file ConvNN.h
    \brief Called by file CovNN.cpp

    No Further Details.
*/

#ifndef __CONVNN_H__
#define __CONVNN_H__

#include "cv.h"
#include "highgui.h"
#include "ml.h"
// #include "_ml.h"
#include "cxcore.h"

using namespace std;

/** \class ConvNN
 *  \brief ConvNN class
 *  see member functions for detail
 */
class ConvNN
{
public:
	ConvNN(void);
	ConvNN(int height, int width, int node, int cNode,
         double alpha, int maxiter);
	~ConvNN(void);

  /** \brief Create CNN models
   * Create CNN models.
   * @param a an integer argument.
   * @param s a constant character pointer.
   * @see publicVar()
   * @return The test results
   */
	void createCNN( );

  /** \brief Save CNN parameters to a file
   * a normal member to save the CNN parameters to a file.
   * @param outFile the name of the output file.
   */
	void writeCNNParams(string outFile);

	/** \brief CNN models
   * a public variable.
   * Saved the CNN model in this variable.
   */
	CvCNNStatModel * m_cnn ;	
	CvCNNStatModel * cnn_train;
  int            m_clipHeight, m_clipWidth;
  int            m_nNode, m_connectNode;
  int            m_max_iter;
	double         m_learningRate;

  /** \brief Load CNN parameters from a file
   * a normal member loading the parameters of CNN from a file.
   * @param inFile the file containing the CNN parameter info.
   */
	void LoadCNNParams(string inFile);

  /** \brief Train a CNN model
   * a normal member to train the CNN.
   * @param trainingData an integer argument.
   * @param responseMat a constant character pointer.
   */
	void trainNN(CvMat *trainingData, CvMat *responseMat,
               CvMat *testingData, CvMat *expectedMat);

  void predictNN(CvMat * inputData, CvMat ** resultData);
};


class CNNIO
{
  enum {NNODE=10};
public:
	CNNIO(void);
	~CNNIO(void);

	void init(int outNode, int width, int height, ConvNN *CNN);

	CvMat** output;
	CvMat * fpredict;
};

#endif // __CONVNN_H__

