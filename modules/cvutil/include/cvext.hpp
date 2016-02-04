/**
 * @file   cvext.h
 * @author Liangfu Chen <chenclf@gmail.com>
 * 
 * @brief  addtional functions for programming using OpenCV
 *
 * Modifications:
 *
 * ================  15:20 2012-11-8  ================
 * 1. CvAbstractTracker class
 * 2. CvActiveContour class
 */

#ifndef __CV_EXT_HPP__
#define __CV_EXT_HPP__

#include "cvext_c.h"
#include <vector>
#include <list>
#include <algorithm>

//-------------------------------------------------------
// CROSS-PLATFORM TIMER
//-------------------------------------------------------
#include "cvtimer.h"

/** 
 * quantize rows of vectors into K classes 
 *  Example:
 *    M=200; N=2; X = randn(M,N); K = 3; Tol = 1e-12;
 *    [codebook,class,quantization_error] = vector_quantization(X,K,Tol)
 * 
 * @param X             IN: MxN CV_32FC1 , indicates input data
 * @param K             IN: classify input data into K clusters
 * @param matCodebook   OUT: size of NxK, type of float matrix
 * @param _class        OUT: size of 1xM, type of uchar matrix
 * @param quant_errs    OUT: errors in each iteration 
 * @param criteria      IN:  iterate termination criteria
 */
CVAPI(void) cvVectorQuantization(
     const CvMat * const X,// input data in rows
     const int K,          // K rows of output data
     CvMat *& matCodebook, // size: NxK, type: float
     CvMat *& _class,      // size: 1xM, type: uchar
     std::vector<float>& quant_errs, // errors in each iter
     const CvTermCriteria criteria); // iter criteria

/**
 * viterbi implementation for online inference with static HMM parameters
 * 
 */
class CV_EXPORTS CvViterbiOnline
{
  bool m_initialized;
  
  int m_nclusters;
  int m_nstates;
  
  // HMM parameters
  CvMat * m_matPi;    // initial probability pi
  CvMat * m_matTrans; // matrix A
  CvMat * m_matEmit;  // matrix B

  CvMat * m_matT1; // probability of the most likely path so far
  CvMat * m_matT2; // KxT states of most likely path so far
  
 public:
  CvViterbiOnline():
      m_initialized(false),
      m_nclusters(0),
      m_nstates(0),
      m_matPi(NULL),
      m_matTrans(NULL),
      m_matEmit(NULL),
      m_matT1(NULL),
      m_matT2(NULL)
  {
  }
  ~CvViterbiOnline()
  {
    cvReleaseMatEx(m_matPi);
    cvReleaseMatEx(m_matTrans);
    cvReleaseMatEx(m_matEmit);
    cvReleaseMatEx(m_matT1);
    cvReleaseMatEx(m_matT2);
  }

  inline bool initialized(){return m_initialized;}

  /** 
   * initialize online viterbi algo. with static HMM parameters
   * 
   * @param pi  IN: initial probability
   * @param A   IN: transition probability
   * @param B   IN: emission probability
   * @param obs IN: first observation
   */
  void init(const CvArr * pi, const CvArr * A, const CvArr * B, int obs)
  {
    CvMat matPiHeader, matTransHeader, matEmitHeader;
    m_matPi = cvCloneMat(cvGetMat(pi, &matPiHeader));
    m_matTrans = cvCloneMat(cvGetMat(A, &matTransHeader));
    m_matEmit = cvCloneMat(cvGetMat(B, &matEmitHeader));

    m_nstates = m_matPi->cols;
    assert(m_nstates==m_matTrans->cols);
    assert(m_nstates==m_matTrans->rows);
    assert(m_nstates==m_matEmit->rows);
    
    m_nclusters = m_matEmit->cols;

    cvLog(m_matPi, m_matPi);
    cvLog(m_matTrans, m_matTrans);
    cvLog(m_matEmit, m_matEmit);

    m_matT1 = cvCreateMat(m_nstates, 1, CV_32FC1); cvZero(m_matT1);
    m_matT2 = cvCreateMat(m_nstates, 1, CV_32SC1); cvZero(m_matT2);

    for (int i = 0; i < m_nstates; i++){
      m_matT1->data.fl[i] =
          m_matPi->data.fl[i]+CV_MAT_ELEM(*m_matEmit, float, i, obs);
    }

    // initialize first column of T2
    for (int i = 0; i < m_nstates; i++){
      CV_MAT_ELEM(*m_matT2, int, i, 0) = i;
    }

    m_initialized = true;
  }

  /** 
   * Inference with static HMM parameter from given single observation.
   *
   * Log-likelyhood used to prevent vanishing floating-point value of
   * probability.
   * 
   * @param obs IN: input observation and update sequence
   * 
   * @return OUT: output optimal state
   */
  int update(int obs)
  {
    int retval = -1;
    assert(m_initialized); // assuming initialized

    CvMat * matTemp =
        cvCreateMat(m_matTrans->rows, m_matTrans->cols, CV_32FC1);
    cvZero(matTemp);
    for (int j = 0; j < m_matTrans->cols; j++){
      for (int k = 0; k < m_matTrans->rows; k++){
        CV_MAT_ELEM(*matTemp, float, k, j) =
            m_matT1->data.fl[k]+CV_MAT_ELEM(*m_matTrans, float, k, j);
      }
    } // finnish
      
    // for each column of transition probability matrix
    CvMat * matIdx = cvCreateMat(1, m_nstates, CV_32SC1);
    for (int i = 0; i < m_matTrans->cols; i++){
      float maxval = -HUGE_VAL; int maxloc = 0;
      for (int j = 0; j < m_matTrans->rows; j++){
        if (CV_MAT_ELEM(*matTemp, float, j, i)>maxval){
          maxval = CV_MAT_ELEM(*matTemp, float, j, i);
          maxloc = j;
        }
      }
      m_matT1->data.fl[i] = maxval;
      matIdx->data.i[i] = maxloc;
    }

	// add current observation
    for (int i = 0; i < m_nstates; i++){
      m_matT1->data.fl[i] =
          m_matT1->data.fl[i] + CV_MAT_ELEM(*m_matEmit, float, i, obs);
    }

    double minval, maxval; CvPoint maxloc;
    cvMinMaxLoc(m_matT1, &minval, &maxval, NULL, &maxloc);
    retval = maxloc.y;
#ifdef _DEBUG
    if (0)
    {
      fprintf(stderr, "%.2f, %.2f\n",
              m_matT1->data.fl[1], m_matT1->data.fl[3]);
    }
#endif
    cvReleaseMatEx(matTemp);
    cvReleaseMatEx(matIdx);
    return retval;
  }
};

class CV_EXPORTS CvActiveContour
{
protected:
  IplImage * m_grayImage;       // original grayscale image
  IplImage * m_gradImage;       // CV_32FC2 gradient of grayscale image
  IplImage * m_gradImageX;      // CV_32FC1 first channel of gradient image
  IplImage * m_gradImageY;      // CV_32FC1 second channel ...
  IplImage * m_magImage;        // magnitude of gradient
  IplImage * m_del2Image;       // laplacian(I) = divergence(gradient(I))

  CvMat * m_shapePrior;         // initial boundary of active contour
public:
  CvActiveContour():
      m_grayImage(NULL),        // CV_8U
      m_gradImage(NULL),        // [grad_x, grad_y] = gradient(I);
      m_gradImageX(NULL),       // grad_x
      m_gradImageY(NULL),       // grad_y
      m_magImage(NULL),         // magnitude of gradient of image
      m_del2Image(NULL),        // laplacian
      m_shapePrior(NULL)        
      {}
  ~CvActiveContour(){
    cvReleaseImageEx(m_grayImage);
  }

  CVStatus initialize(
      const CvArr * img,
      const CvArr * shape CV_DEFAULT(NULL)
                      )
  {
    return 1;
  }

  CVStatus deform(
      const float alpha,
      const float beta,
      const float gamma
                 )
  {
    return 1;
  }

  // virtual void interp()=0;
};

/** 
 * 
 * 
 * @param pImage in: input image
 * @param vBlobCoordinates out: list of blob coordinates
 */
CV_INLINE
void cvFindBlobs(IplImage* pImage,
				 std::vector< std::vector<CvPoint> >& vBlobCoordinates)
{
  int m_nImageH = pImage->height;
  int m_nImageW = pImage->widthStep;
	
  uchar **data1 = new uchar* [m_nImageH];
  unsigned short **data2 = new unsigned short * [m_nImageH];
	
  unsigned short *pBlobImage = new unsigned short [m_nImageW*m_nImageH];
  memset(pBlobImage, 0, m_nImageW*m_nImageH*sizeof(unsigned short));
	
  uchar **pTemp1 = data1;
  uchar  *pSrc1  = (uchar*)pImage->imageData;
	
  unsigned short **pTemp2 = data2;
  unsigned short  *pSrc2  = pBlobImage;
	
  for (int i = 0; i < m_nImageH; i++)
  {
    *(pTemp1++) = pSrc1;
    pSrc1 += m_nImageW;
		
    *(pTemp2++) = pSrc2;
    pSrc2 += m_nImageW;
  }
	
  int Xdim = m_nImageW;
  int Ydim = m_nImageH;
	
  const int SIZE = 75000;
	
  int parent[SIZE] = { 0 }; 
	
  int Color = 1;
  int Count = 0;
	
  int i (0), x (0), y (0), c1 (0), c2 (0); 
	
  for (i = 0; i < SIZE; i++)
  {
    parent[i] = 0;
  }
	
  //  Mark the blobs in the input image 
  for (y = 0; y < Ydim; y++)
    for (x = 0; x < Xdim; x++)
    {
      if (data1[y][x] > 0)
      {
        // Handle merger of two blobs 
        if ((x > 0) && (data2[y][x-1] > 0) &&
            (y > 0) && (data2[y-1][x] > 0) &&
            (data2[y][x-1] != data2[y-1][x]))
        {
          // Make one parent point to the other 
          c1 = data2[y][x-1];
          c2 = data2[y-1][x];
          while (parent[c1] > 0)
            c1 = parent[c1];
          while (parent[c2] > 0)
            c2 = parent[c2];
          if (c1 != c2)
            parent[c1] = c2;
					
          // Mark pixel with blob color 
          c1 = data2[y][x-1];
          data2[y][x] = c1;
        }
        // Handle blob to the left 
        else if ((x > 0) && (data2[y][x-1] > 0))
        {
          c1 = data2[y][x-1];
          data2[y][x] = c1;
        }
				
        // Handle blob to the left uppper diagonal
        else if ((x > 0) && (y > 0) && (data2[y-1][x-1] > 0))
        {
          c1 = data2[y-1][x-1];
          data2[y][x] = c1;
        }
				
        // Handle blob above 
        else if ((y > 0) && (data2[y-1][x] > 0))
        {
          c1 = data2[y-1][x];
          data2[y][x] = c1;
        }
				
        // Handle new blob 
        else
        {
          parent[Color] = -1;
          data2[y][x] = Color++;
					
          if (Color >= SIZE)
          {
            printf("Error: too many blobs detected \n");
            break;
          }
        }
      }
      else
      {
        // Handle merger of two blobs 
        if ((x > 0) && (data2[y][x-1] > 0) &&
            (y > 0) && (data2[y-1][x] > 0) &&
            (data2[y][x-1] != data2[y-1][x]))
        {
          // Make one parent point to the other 
          c1 = data2[y][x-1];
          c2 = data2[y-1][x];
          while (parent[c1] > 0)
            c1 = parent[c1];
          while (parent[c2] > 0)
            c2 = parent[c2];
          if (c1 != c2)
            parent[c1] = c2;   
        }
      }
			
      if (Color >= SIZE)
      {
        printf("Error: too many blobs detected \n");
        break;
      }
    }
	
  Count = -1;
	
  for (c1 = 1; c1 < Color; c1++)
  {
    if (parent[c1] < 0)
      parent[c1] = Count--;
  }
	
  CvPoint point;
  for (y = 0; y < Ydim; y++)
    for (x = 0; x < Xdim; x++)
    {
      c1 = data2[y][x];
			
      while (parent[c1] > 0)
        c1 = parent[c1];
			
      if (-parent[c1])
      {
        point.x = x;
        point.y = y;
        if (-parent[c1] >= (int)vBlobCoordinates.size())
        {
          vBlobCoordinates.resize(-parent[c1]+1);
        }
        vBlobCoordinates[-parent[c1]].push_back(point);
      }
    }
	
  int nNumberOfBlobs = -(Count + 1);
  fprintf(stderr, "Region finding located %d connected regions.\n",
          nNumberOfBlobs);
    
  delete [] data1;
  delete [] data2;
  delete [] pBlobImage;
}

#endif //__CV_EXT_HPP__
