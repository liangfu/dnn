/**
 * @file   cvsparsecoding.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Jul 19 16:31:08 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_SPARSE_CODING_H__
#define __CV_SPARSE_CODING_H__

#include "cvext_c.h"

/** 
 * extract dictionary from grayscale image
 * 
 * @param img  in:  grayscale image 
 * @param dict out: NxM matrix to represent dictionary
 * 
 * @return status code
 */
int icvDictDenseSample(CvMat * img, CvMat * dict);

/** 
 * shuffle dictionary elements
 * 
 * @param dict in: dictionary ordered as NxM matrix
 * 
 * @return status code
 */
int icvDictShuffle(CvMat * dict);

/** 
 * display dictionary 
 * 
 * @param dict in: extracted dictionary code to display
 * 
 * @return status code
 */
#ifndef ANDROID
int icvDictShow(CvMat * dict);
#endif // ANDROID

/** 
 * perform matching pursuit algorithm upon dictionary D and signal y,
 * and output sparse matrix x as sparse representation
 * 
 * @param D in:  dictionary ordered as N-by-M dimension
 * @param y in:  input signal to be estimated
 * @param x out: sparse matrix 
 * @param maxiter in: maximum number of iterations 
 * @param epsilon in: L2-norm error - criteria for termination
 * 
 * @return status code
 */
int icvOrthogonalMatchingPursuit(CvMat * D, CvMat * y, CvMat * x,
                                 int maxiter=10, float epsilon=.02f);

/** 
 * perform basis pursuit with primal-dual log-barrier algorithm
 * for l1-norm minimization
 * 
 * @param A in:  MxN dictionary 
 * @param s in:  Mx1 signal
 * @param x out: Nx1 coefficient
 * @param maxiter in: termination criteria
 * 
 * @return 
 */
int icvBasisPursuit(CvMat * A, CvMat * s, CvMat * x, int maxiter);
int icvBasisPursuit_optimized(CvMat * A, CvMat * b, CvMat * x);

/**
 * abstract class for learning sparse representation
 * from a list of static image files
 */
class CvSparseLearner
{
public:
  CvSparseLearner(){}
  ~CvSparseLearner(){}

#ifndef ANDROID
  int learn(const char * filelist[], int nfiles);//{return 1;}
#endif // ANDROID
};

class CV_EXPORTS CvSparseCoding
{
  CvSparseLearner m_learner;
public:
  CvSparseCoding(){}
  ~CvSparseCoding(){}
};

#endif // __CV_SPARSE_CODING_H__
