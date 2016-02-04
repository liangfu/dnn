/**
 * @file   cvsort.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Dec 11 13:45:20 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"

/** 
 * Calculate gradient of source image.
 * In case of destinate image is CV_8UC1, the result would be
 *     dst = dilate(src) - erode(src);
 * In case of destinate image is CV_32FC3, 
 *     the function treat the given source image as a height map.
 *     therefore, result in 3D direction of original normal vectors
 *
 * @param src       IN:     original gray image as input
 * @param dst       OUT:    CV_32FC3 or CV_8U output gradient map
 * @param ksize     IN:     default value is set to 1,
 *                          indicating the kernel size in morphing function
 */
void cvCalcGradient(const CvArr * src,
                    CvArr * dst,
                    const int ksize )
{
//  CV_FUNCNAME("cvCalcGradient");
//  IplImage grayHeader, gradHeader;
//  __BEGIN__;
//  IplImage * gray = cvGetImage(src, &grayHeader);
//  IplImage * grad = cvGetImage(dst, &gradHeader);
//  CV_ASSERT(gray->height==grad->height && gray->width==grad->width);
//  // assume 8-bit single channel image as input
//  //CV_ASSERT(cvGetElemType(gray)==CV_8U);  
//  
//  //if (grad->nChannels==1)
//  if (cvGetElemType(grad)==CV_8U || 
//	  cvGetElemType(grad)==CV_32F)
//  {
//    CV_ASSERT(ksize==1);
//	CV_ASSERT(grad->nChannels==1);
//#if 0
//    // 0  -1   0
//    //-1   0   1
//    // 0   1   0
//    float vMagKernel[3][3] = {{0,-1,0},{-1,0,1},{0,1,0}};
//    CvMat MagGradKernel =
//        cvMat(ksize*2+1, ksize*2+1, CV_32F, vMagKernel);
//    cvFilter2D(gray, grad, &MagGradKernel,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//#else
//    // 0   0   0
//    //-1   0   1
//    // 0   0   0
//    float vKernelX[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
//    // 0  -1   0
//    // 0   0   0
//    // 0   1   0
//    float vKernelY[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};
//    IplImage * grad_x =
//        cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
//    IplImage * grad_y =
//        cvCreateImage(cvGetSize(src), IPL_DEPTH_32F, 1);
//    CvMat GradKernelX =
//        cvMat(ksize*2+1, ksize*2+1, CV_32F, vKernelX);
//    CvMat GradKernelY =
//        cvMat(ksize*2+1, ksize*2+1, CV_32F, vKernelY);
//    cvFilter2D(gray, grad_x, &GradKernelX,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    cvFilter2D(gray, grad_y, &GradKernelY,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    //cvMerge(grad_x, grad_y, NULL, NULL, grad);
//	cvCartToPolar(grad_x,grad_y, grad);
//    cvReleaseImageEx(grad_x);
//    cvReleaseImageEx(grad_y);
//#endif
//  }else if (cvGetElemType(grad)==CV_8UC2){
//    CV_ASSERT(ksize==1);
//    // 0   0   0
//    //-1   0   1
//    // 0   0   0
//    float vKernelX[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
//    // 0  -1   0
//    // 0   0   0
//    // 0   1   0
//    float vKernelY[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};
//    IplImage * grad_x =
//        cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
//    IplImage * grad_y =
//        cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 1);
//    CvMat GradKernelX =
//        cvMat(ksize*2+1, ksize*2+1, CV_32F, vKernelX);
//    CvMat GradKernelY =
//        cvMat(ksize*2+1, ksize*2+1, CV_32F, vKernelY);
//    cvFilter2D(gray, grad_x, &GradKernelX,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    cvFilter2D(gray, grad_y, &GradKernelY,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    cvMerge(grad_x, grad_y, NULL, NULL, grad);
//    cvReleaseImageEx(grad_x);
//    cvReleaseImageEx(grad_y);
//  }else if (cvGetElemType(grad)==CV_32FC2 ||
//            cvGetElemType(grad)==CV_64FC2){
//    CV_ASSERT(ksize==1);
//    // 0   0   0
//    //-1   0   1
//    // 0   0   0
//    float vKernelX[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
//    // 0  -1   0
//    // 0   0   0
//    // 0   1   0
//    float vKernelY[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};
//    int localtype;
//    if (cvGetElemType(grad)==CV_32FC2)
//      localtype=CV_32F;
//    else
//      localtype=CV_64F;
//    CvMat * grad_x = cvCreateMat(grad->height, grad->width, localtype);
//    CvMat * grad_y = cvCreateMat(grad->height, grad->width, localtype);
//    CvMat GradKernelX =
//        cvMat(ksize*2+1, ksize*2+1, localtype, vKernelX);
//    CvMat GradKernelY =
//        cvMat(ksize*2+1, ksize*2+1, localtype, vKernelY);
//    cvFilter2D(gray, grad_x, &GradKernelX,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    cvFilter2D(gray, grad_y, &GradKernelY,
//               cvPoint(ksize, ksize) // archor of kernel
//               );
//    cvMerge(grad_x, grad_y, NULL, NULL, grad);
//    cvReleaseMatEx(grad_x);
//    cvReleaseMatEx(grad_y);
//  }else{
//    fprintf(stderr, "Warning: data type not supported !\n");
//  }
//  __END__;
}

//template <typename DataType> 
//void cvCalcGradient(const CvArr * _src,
//                    CvArr * _dx ,
//                    CvArr * _dy ,
//                    CvArr * _mag ) // magnitude (optional)
//{
//  CV_FUNCNAME("cvCalcGradient2");
//  CvMat srchdr, dxhdr, dyhdr, maghdr;
//  CvMat * src=NULL, * dx=NULL, * dy=NULL, * mag=NULL;
//  src = cvGetMat(_src, &srchdr);
//  int type = CV_MAT_TYPE(src->type);
//  int ncols = src->cols;  // width
//  int nrows = src->rows;  // height
//
//  if (_dx) {dx = cvGetMat(_dx, &dxhdr);}
//  else {dx = cvCreateMat(nrows, ncols, type);}
//  if (_dy) {dy = cvGetMat(_dy, &dyhdr);}
//  else {dy = cvCreateMat(nrows, ncols, type);}
//  if (_mag) {mag = cvGetMat(_mag, &maghdr);}
//
//  __BEGIN__;
//
//  assert( (type==CV_32F)||(type==CV_64F)||(type==CV_8U)||(type==CV_32S) );
//  assert(type==CV_MAT_TYPE(dx->type));
//  assert(type==CV_MAT_TYPE(dy->type));
//  if (_mag) {assert(type==CV_MAT_TYPE(mag->type));}
//
//  const int ksize = 1;
//  // float xkernel_data[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
//  // float ykernel_data[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};
//  DataType xkernel_data[3][3] = {{0,0,0},{-1,0,1},{0,0,0}};
//  DataType ykernel_data[3][3] = {{0,-1,0},{0,0,0},{0,1,0}};
//  CvMat xkernel = cvMat(3,3, type, xkernel_data);
//  CvMat ykernel = cvMat(3,3, type, ykernel_data);
//  cvFilter2D(src, dx, &xkernel, cvPoint(ksize, ksize));
//  cvFilter2D(src, dy, &ykernel, cvPoint(ksize, ksize));
//
//  if (mag) {cvCartToPolar(dx,dy,mag);}
//
//  if (_dx==NULL) {cvReleaseMatEx(dx);}
//  if (_dy==NULL) {cvReleaseMatEx(dy);}
//  __END__;
//}

/** 
 * OpenCV 2.x provide cvSort function for sorting array with index values
 * optionally returned. 
 * 
 * @param src IN: CV_32FC1 supported ONLY, single row with multiple columns
 * @param dst OUT: same as input
 * @param idx OUT: CV_32SC1 index position of original array. (Optional)
 */
//#if CV_MAJOR_VERSION==1 && !defined(ANDROID) && !defined(CV_SORT_DESCENDING)
//CV_INLINE
//bool icvSortCmpFuncAsc(const std::pair<float, int> & p1,
//                       const std::pair<float, int> & p2)
//{return p1.first<p2.first;}
//CV_INLINE
//bool icvSortCmpFuncDesc(const std::pair<float, int> & p1,
//                        const std::pair<float, int> & p2)
//{return p1.first>p2.first;}
//void cvSort(const CvArr * src, CvArr * dst, CvArr * idx ,
//            const int order )
//{
//  CV_FUNCNAME("cvSort");
//  CvMat srcheader, dstheader, idxheader;
//  CvMat * matSrc = cvGetMat(src, &srcheader);
//  CvMat * matDst = cvGetMat(dst, &dstheader);
//  CvMat * matIdx = NULL;
//  std::vector<std::pair<float, int> > data(matSrc->cols);
//  __BEGIN__;
//  if (idx!=NULL){ matIdx = cvGetMat(idx, &idxheader); }
//
//  // assume data on the columns ONLY
//  CV_ASSERT(matSrc->rows==1); CV_ASSERT(matDst->rows==1);
//  CV_ASSERT(matSrc->cols==matDst->cols);
//  CV_ASSERT(matSrc->cols==matIdx->cols);
//  CV_ASSERT(cvGetElemType(matSrc)==cvGetElemType(matDst));
//
//  CV_ASSERT(cvGetElemType(matSrc)==CV_32FC1);
//  CV_ASSERT(cvGetElemType(matDst)==CV_32FC1);
//  if (idx!=NULL){ CV_ASSERT(cvGetElemType(matIdx)==CV_32SC1); }
//
//  for (int i = 0; i < matSrc->cols; i++){
//    // data[i].<first,second> ---- <value, idx>
//    data[i] = std::make_pair<float, int>(matSrc->data.fl[i], i);
//  }
//  if (order==CV_SORT_ASCE){
//    std::sort(data.begin(), data.end(), icvSortCmpFuncAsc);
//  }else{
//    std::sort(data.begin(), data.end(), icvSortCmpFuncDesc);
//  }
//  for (int i = 0; i < matSrc->cols; i++){
//    matDst->data.fl[i] = data[i].first;
//    if (idx!=NULL){ matIdx->data.i[i]  = data[i].second; } // output index
//  }  
//  __END__;
//}
//#endif // CV_MAJOR_VERSION==1

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
void cvVectorQuantization(const CvMat * const X,// input data in rows
                          const int K,          // K rows of output data
                          CvMat *& matCodebook, // size: NxK, type: float
                          CvMat *& _class,      // size: 1xM, type: uchar
						  std::vector<float>& quant_errs, // errors in each iter
                          const CvTermCriteria criteria // iter criteria
                          )
{
  assert(cvGetElemType(X)==CV_32FC1);
  CvSize size = cvGetSize(X);
  int M = size.height;
  int N = size.width;

  float quant_errs_[2] = {10.f, 1.f};
  // CvMat quant_errs = cvMat(2, 1, CV_32FC1, quant_errs_);
  quant_errs.clear();
  quant_errs.resize(2);
  quant_errs[0] = quant_errs_[0];
  quant_errs[1] = quant_errs_[1];

  CvMat * matX = cvCloneMat(X);
  //CvMat * matCodebook = cvCreateMat(K, N, CV_32FC1);
  assert(matCodebook->rows==K);
  assert(matCodebook->cols==N);
  assert(cvGetElemType(matCodebook)==CV_32FC1);

  /*********************************************************
   * pick sample point at random
   ********************************************************/
  {
	std::vector<int> vTmp(M);
    for (int i = 0; i < M; i++){vTmp[i] = i;}
	std::random_shuffle(vTmp.begin(), vTmp.end());

    // randomly get first K elements in X
    for (int i = 0; i < K; i++){
      for (int j = 0; j < N; j++){
        CV_MAT_ELEM(*matCodebook, float, i, j) =
            CV_MAT_ELEM(*X, float, vTmp[i], j);
      }
    }
  }

  for (int iter = 0; iter < criteria.max_iter; iter++)
  {
    CvMat * matCodebook_new =
        cvCreateMat(K, N, CV_32FC1); // store code book
    cvZero(matCodebook_new);
    CvMat * matClass =
        cvCreateMat(M, 1, CV_8UC1); // store classification info
    cvZero(matClass);
    CvMat * matMembership =
        cvCreateMat(M, K, CV_8UC1);
    cvZero(matMembership);
    double quant_err = 0.;

    /*********************************************************
     * move the chosen centroid towards the sample point
     * by small fraction of the distance
     ********************************************************/
    // 
    for (int m = 0; m < M; m++){
      CvMat * matDist =
          cvCreateMat(K, 1, CV_32FC1);

      for (int k = 0; k < K; k++)
      {
        // dist(k) = sum((X(m,:), codebook(k,:).^2));
        CvMat matSubXHeader;
        CvMat * matSubX = cvGetSubRect(X, &matSubXHeader,
                                       cvRect(0, m, N, 1));
        CvMat matSubCodebookHeader;
        CvMat * matSubCodebook =
            cvGetSubRect(matCodebook, &matSubCodebookHeader,
                         cvRect(0, k, N, 1));
        CvMat * matTmp =
            cvCreateMat(1, N, CV_32FC1);
        CvMat * matTmpRes =
            cvCreateMat(1, N, CV_32FC1);
        cvSub(matSubX, matSubCodebook, matTmp);

        // per-element product of two arrays
        cvMul(matTmp, matTmp, matTmpRes); 
        CV_MAT_ELEM(*matDist, float, k, 0) = cvSum(matTmpRes).val[0];
        cvReleaseMatEx(matTmp);
        cvReleaseMatEx(matTmpRes);
      }
      // vector<float> vDist = cvMat2STLvectorf(matDist, 1);
      
      double min_dist = 0.; double max_dist = 0.;
      CvPoint k_min;
      cvMinMaxLoc(matDist, &min_dist, &max_dist, &k_min, NULL);

      int km = k_min.y; 
	  if (km<0){km=0;} // MATLAB computes one from multiple minima
      CV_MAT_ELEM(*matClass, uchar, m, 0) = km;
      CV_MAT_ELEM(*matMembership, uchar, m, km) = 1;

      for (int i = 0; i < N; i++){
        CV_MAT_ELEM(*matCodebook_new, float, km, i) =
            (CV_MAT_ELEM(*matCodebook_new, float, km, i)+
             CV_MAT_ELEM(*X, float, m, i));
      }

      quant_err += min_dist;
	  cvReleaseMatEx(matDist);
    }
	cvCopy(matClass, _class);
#ifdef _DEBUG
	{
      //vector<uchar> vClass = cvMat2STLvectorb(_class, 1);
	  int sum_of_membership = cvSum(matMembership).val[0];
      // all of the M elements are classified
	  assert(sum_of_membership==M);  
	}
#endif
    /*********************************************************
     ***** set the chosen centroid's sensitivity to zero *****
     ********************************************************/
    // for each chosen element
    for (int k = 0; k < K; k++){
      CvMat matSubMembershipHeader;
      CvMat * matSubMembership =
          cvGetSubRect(matMembership, &matSubMembershipHeader,
                       cvRect(k, 0, 1, M));
      int no_of_elements =
          cvSum(matSubMembership).val[0];

      if (no_of_elements>1){
        for (int i = 0; i < N; i++){
          CV_MAT_ELEM(*matCodebook_new, float, k, i) =
              (CV_MAT_ELEM(*matCodebook_new, float, k, i)/no_of_elements);
        }
      }else if (no_of_elements==0){
        // if the class represented by a codevector has no elements
        CvMat matSubCodebookHeader_new;
        CvMat * matSubCodebook_new  =
            cvGetSubRect(matCodebook_new, &matSubCodebookHeader_new,
                         cvRect(0, 0, N, k+1));

        for (int i = 0; i < N; i++){
          float sumval = 0.0f;
          for (int j = 0; j < k+1; j++){
            sumval += CV_MAT_ELEM(*matCodebook_new, float, j, i);
          }
          float avgval = sumval/(k+1);
          CV_MAT_ELEM(*matCodebook_new, float, k, i) = avgval;
        }
      }
    }

    cvCopy(matCodebook_new, matCodebook);
    quant_err = quant_err/M;
    quant_errs.push_back(quant_err);

    int quant_errs_size = quant_errs.size();
    if (abs(quant_errs[quant_errs_size-1]-
            quant_errs[quant_errs_size-2])<criteria.epsilon)
    {
      // collect classification info before end of iteration
      assert(matClass->cols==_class->cols);
      assert(matClass->rows==_class->rows);
      assert(cvGetElemType(matClass)==cvGetElemType(_class));
      cvCopy(matClass, _class);
      cvReleaseMatEx(matCodebook_new);
      cvReleaseMatEx(matClass);
      cvReleaseMatEx(matMembership);
      break;
    }
      
    if (iter+1==criteria.max_iter)
    {
      // collect classification info before end of iteration
      assert(matClass->cols==_class->cols);
      assert(matClass->rows==_class->rows);
      assert(cvGetElemType(matClass)==cvGetElemType(_class));
      cvCopy(matClass, _class);
    }

    cvReleaseMatEx(matCodebook_new);
    cvReleaseMatEx(matClass);
    cvReleaseMatEx(matMembership);
  }// end of maxiter

  cvReleaseMatEx(matX);
  //cvReleaseMatEx(matCodebook);
}
