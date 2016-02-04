/**
 * @file   cvext_motion.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Dec 05 17:01:26 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvext_c.h"

/** 
 * continuously track a set of points in image
 * 
 * @param curr IN: 
 * @param next IN:
 * @param points IN:OUT:
 *               CV_32SC2 landmark points to track using optical flow
 * 
 * @return OUT: error code
 */
CVStatus cvOpticalFlowPointTrack(
    const CvArr * curr,                    // current grayscale frame
    const CvArr * next,                    // next frame
    CvArr * points,                        // CV_32SC2 points to track
    const CvSize winsize,
    const int nlevels                      // number of pyramid levels
                                 )
{
  CV_FUNCNAME("cvOpticalFlowPointTrack");
  assert(cvGetElemType(curr)==CV_8U);
  assert(cvGetElemType(next)==CV_8U);
  assert(cvGetElemType(points)==CV_32FC2);

  IplImage currheader, nextheader; CvMat ptsheader;
  
  static IplImage * buffImageL = NULL;  // WARNING: memory leak !!!
  static IplImage * buffImageR = NULL;
  CvSize imsize = cvGetSize(curr);
  IplImage * currImage = cvGetImage(curr, &currheader);
  IplImage * nextImage = cvGetImage(next, &nextheader);
  CvMat * contour =  cvGetMat(points, &ptsheader);
  CvMat * contourdesc =
      cvCreateMat(1, contour->cols, CV_32FC2);
  CvMat * contourdescOUT =
      cvCreateMat(1, contour->cols, CV_32FC2);
  // cvConvert(contour, contourdesc);
  cvCopy(contour, contourdesc);
  char * status = new char[contour->cols]; 
  float * track_err = new float[contour->cols];

  __BEGIN__;
#if 0
  if (buffImageL==NULL || buffImageR==NULL){
    buffImageL = 
        cvCreateImage(cvSize(imsize.width+8, imsize.height/3),
                      IPL_DEPTH_32F, 1);
    buffImageR = 
        cvCreateImage(cvSize(imsize.width+8, imsize.height/3),
                      IPL_DEPTH_32F, 1);
    cvCalcOpticalFlowPyrLK(currImage, nextImage,
                           buffImageL, buffImageR, // pyramid buffer
                           (CvPoint2D32f*)contourdesc->data.ptr,
                           (CvPoint2D32f*)contourdescOUT->data.ptr,
                           contour->cols, // number of points
                           winsize, // window size
                           nlevels,       // number of levels for pyramid
                           status,  // whether points are found
                           track_err,  // tracking error of each point
                           cvTermCriteria(CV_TERMCRIT_EPS+
                                          CV_TERMCRIT_ITER,
                                          30, 0.01 ),
                           0 // compute pyramid for both image
                           );
  }else{
    cvCopy(buffImageR, buffImageL);
    cvCalcOpticalFlowPyrLK(currImage, nextImage,
                           buffImageL, buffImageR, // pyramid buffer
                           (CvPoint2D32f*)contourdesc->data.ptr,
                           (CvPoint2D32f*)contourdescOUT->data.ptr,
                           contour->cols, // number of points
                           winsize, // window size
                           nlevels,       // number of levels for pyramid
                           status,  // whether points are found
                           track_err,  // tracking error of each point
                           cvTermCriteria(CV_TERMCRIT_EPS+
                                          CV_TERMCRIT_ITER,
                                          30, 0.01 ),
                           CV_LKFLOW_PYR_A_READY
                           );
  }
  // cvConvert(contourdescOUT, contour);
  for (int ff = 0; ff < contour->cols; ff++){
    if (status[ff]){
      // CV_MAT_ELEM(*contour, CvPoint, 0, ff) =
      //     cvPointFrom32f(
      //         CV_MAT_ELEM(*contourdescOUT, CvPoint2D32f, 0, ff));
      CV_MAT_ELEM(*contour, CvPoint2D32f, 0, ff) =
          CV_MAT_ELEM(*contourdescOUT, CvPoint2D32f, 0, ff);
    }else{
      CV_MAT_ELEM(*contour, CvPoint2D32f, 0, ff) =
          CV_MAT_ELEM(*contourdesc, CvPoint2D32f, 0, ff);
    }
  }
  delete [] status;
  delete [] track_err;
  cvReleaseMatEx(contourdesc);
  cvReleaseMatEx(contourdescOUT);
#endif
  __END__;
  return CV_StsOk;
}

// image alignment
////////////////////////////////////////////////////////////////
////////////////// MATLAB CODE ////////////////
////////////////////////////////////////////////////////////////
// function fit = affine_fa(img, tmplt, p_init, n_iters, verbose, step_size)
// % AFFINE_FA - Affine image alignment using forwards-additive algorithm
// %   FIT = AFFINE_FA(IMG, TMPLT, P_INIT, N_ITERS, VERBOSE)
// %   Align the template image TMPLT to an example image IMG using an
// %   affine warp initialised using P_INIT. Iterate for N_ITERS iterations.
// %   To display the fit graphically set VERBOSE non-zero.
// %
// %   p_init = [p1, p3, p5
// %             p2, p4, p6];
// %
// %   This assumes greyscale images and rectangular templates.
// %
// %   c.f. Lucas-Kanade 
// % Common initialisation
// init_a;

// % Pre-computable things ----------------------------------------------
// % 3a) Compute image gradients - will warp these images in step 3b)
// [img_dx img_dy] = gradient(img);
// % 4) Evaluate Jacobian - constant for affine warps, but not in general
// dW_dp = jacobian_a(w, h);

// % Lucas-Kanade, Forwards Additive Algorithm --------------------------
// for f=1:n_iters
// 	% 1) Compute warped image with current parameters
// 	IWxp = warp_a(img, warp_p, tmplt_pts);
//  subplot(223),imshow(IWxp,[]);pause(0.1);
// 	% 2) Compute error image
// 	error_img = tmplt - IWxp;
// 	% -- Save current fit parameters --
// 	fit(f).warp_p = warp_p;                               warp_p,
// 	fit(f).rms_error = sqrt(mean(error_img(:) .^2));      

// 	% -- Really iteration 1 is the zeroth, ignore final computation --
// 	if (f == n_iters) break; end             % maxiter
// 	if fit(f).rms_error<1.e-7, break; end    % small error
// 	% 3b) Evaluate gradient
// 	nabla_Ix = warp_a(img_dx, warp_p, tmplt_pts);
// 	nabla_Iy = warp_a(img_dy, warp_p, tmplt_pts);
// 	% 4) Evaluate Jacobian - constant for affine warps. Precomputed above
// 	% 5) Compute steepest descent images, VI_dW_dp
// 	VI_dW_dp = sd_images(dW_dp, nabla_Ix, nabla_Iy, N_p, h, w); 
// 	% 6) Compute Hessian and inverse
// 	H = hessian(VI_dW_dp, N_p, w);
// 	H_inv = inv(H);
// 	% 7) Compute steepest descent parameter updates
// 	sd_delta_p = sd_update(VI_dW_dp, error_img, N_p, w);
// 	% 8) Compute gradient descent parameter updates
// 	delta_p = H_inv * sd_delta_p;
// 	% 9) Update warp parmaters
// 	warp_p = update_step(warp_p, delta_p);
// end

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// function warp_p = update_step(warp_p, delta_p)
// % Compute and apply the additive update
// delta_p = reshape(delta_p, 2, 3);
// warp_p = warp_p + delta_p;
////////////////////////////////////////////////////////////////

/** 
 * Image alignment with template
 * 
 * @param _image 
 * @param _tmplt 
 * @param _init_p 
 * 
 * @return 
 */
CVAPI(CVStatus) cvAlignImage_Affine(const CvArr * _image, 
									const CvArr * _tmplt, 
									CvArr * _init_p,
                                    int method)
{
  // termination criteria
  const int maxiter = 100;
  const double epsilon = 1.e-7;
  
  assert(method==CV_ALIGN_FA); // forward addition
  CvMat imghdr;
  CvMat * imgSrc = cvGetMat(_image, &imghdr);

  const int nrows = imgSrc->rows;
  const int ncols = imgSrc->cols;

  CvMat * dx = cvCreateMat(nrows, ncols, CV_32S);
  CvMat * dy = cvCreateMat(nrows, ncols, CV_32S);
  //{
  //  CvMat * imgSrc32s = cvCreateMat(nrows, ncols, CV_32S);
  //  cvConvert(imgSrc, imgSrc32s);
  //  cvCalcGradient<int>(imgSrc32s, dx, dy);
  //  cvReleaseMatEx(imgSrc32s);
  //}
  //CvMat * dW_dp = cvCreateMat(nrows, ncols, CV_64F);
  // cvCalcJacobian(affine_func, nrows, ncols, dW_dp);

  int iter=0;
  while ( (iter++<=maxiter) ){
    
  }

  //cvReleaseMatEx(dW_dp);
  cvReleaseMatEx(dx);
  cvReleaseMatEx(dy);

  return CV_StsOk;
}


////////////////////////////////////////////////////////////////
// % jacobian for affine warp
// jac_x = kron([0:nx - 1],ones(ny, 1));
// jac_y = kron([0:ny - 1]',ones(1, nx));
// jac_zero = zeros(ny, nx);
// jac_one = ones(ny, nx);
// dW_dp = [jac_x, jac_zero, jac_y, jac_zero, jac_one, jac_zero;
//          jac_zero, jac_x, jac_zero, jac_y, jac_zero, jac_one];
////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////
// % JACOBIAN - compute jacobian matrix with given function and data
// %  s = jacobian(f, x, tol);
// %    f is a multivariable function handle, 
// %    x is a starting point
////////////////
// function s = jacobian(f, x, tol)
//   if nargin == 2
//     tol = 10^(-5);
//   end
//   x = x(:)'; % force row vector
//   while 1
//     % if x and f(x) are row vectors, we need transpose operations here
//     y = x' - jacob(f, x)\f(x)';             % get the next point
//     if norm(f(y))<tol                       % check error tolerate
//       s = y';
//       return;
//     end
//     x = y';
//   end  
// end
////////////////
// function j = jacob(f, x)         % approximately calculate Jacobian matrix
//   k = length(x);
//   j = zeros(k, k);
//   x2 = x;
//   dx = 0.001;
//   for m = 1:k 
//     x2(m) = x(m)+dx;
//     j(m, :) = (f(x2)-f(x))/dx;    % partial derivatives in m-th row
//     x2(m) = x(m);
//   end
// end
////////////////////////////////////////////////////////////////

void cvCalcJacobian(
    void (*jacob_fun)(CvMat * src, CvMat * dst, CvMat * params) ,
    CvMat * data, CvMat * jacobian)
{
  
}

// cvMeanShiftEx
//
// @param src     	input data points, NxD matrix with FLOAT32 data
//                  (CV_32F data type assumed)
// @param dst     	mean shifted data points, same type and size as input
// @param ksize   	kernel size of Gaussian or flat kernel
//                  ( currently ONLY flat kernel supported)
// @param criteria  stop criteria of meanshift iteration
// 
// for both source array and output array, NxD matrix is assumed
// N points as input with D dimensional data
// 
// Usage:
//	const int npoints = 300;
//	const int ndims=2;
//	CvMat orig=cvMat(npoints,ndims,CV_32F, orig_data);
//	CvMat * data = cvCreateMat(orig.rows, orig.cols, CV_32F);
//	cvAddS(&orig, cvScalar(5),&orig);
//	cvScale(&orig,&orig,18);
//CV_TIMER_START();
//	cvMeanShiftEx(&orig, data, 60, cvTermCriteria(3,10,0.1));
//CV_TIMER_SHOW();
//	cvNamedWindow("Test");
//	IplImage * dispImage = cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);
//  cvZero(dispImage);
//	for (int i = 0; i < npoints; i++)
//		cvCircle(dispImage, cvPoint(CV_MAT_ELEM(orig,float,i,0),
//               CV_MAT_ELEM(orig,float,i,1)),1,cvScalar(255),-1);
//	for (int i = 0; i < npoints; i++)
//		cvCircle(dispImage, cvPoint(CV_MAT_ELEM(*data,float,i,0),
//               CV_MAT_ELEM(*data,float,i,1)),1,cvScalar(0,255),-1);
//	cvShowImage("Test", dispImage); cvWaitKey();
//	cvReleaseMat(&data);
//	cvReleaseImage(&dispImage);
//	cvDestroyWindow("Test");
// 
// @see cvMeanShift
void cvMeanShiftEx(const CvArr * _src,
                   CvArr * _dst,            // mean shifted data as output
				   const float ksize,       // kernel size
				   CvTermCriteria criteria) // mean shift iteration criteria
{
  CvMat *src,src_stub,*dst,dst_stub;
  src = cvGetMat(_src, &src_stub);
  dst = cvGetMat(_dst, &dst_stub);
  const int npoints = src->rows;
  const int ndims = src->cols;

  const float epsilon = criteria.epsilon;
  const int maxiter = criteria.max_iter;
  const int type = CV_MAT_TYPE(src->type);

  CvMat * dist, * dist_t, * numerator;
  float * newy; int iter=0;
    
  CV_FUNCNAME("cvCalcMeanShift");
  __BEGIN__;
  CV_ASSERT(src->rows==dst->rows);
  CV_ASSERT(src->cols==dst->cols);
  CV_ASSERT(src->type==dst->type);
  CV_ASSERT(type==CV_32F);     // ONLY FLOAT32 is supported !!
  CV_ASSERT(criteria.type==3); // both max_iter and epsilon
  // save a copy
  cvCopy(src, dst);

  dist = cvCreateMat(npoints, 1, CV_32F);
  dist_t = cvCreateMat(1, npoints, CV_32F);
  numerator = cvCreateMat(1, ndims, CV_32F);
  newy = new float[ndims];

  iter=0;
  for (int ii = 0; ii < npoints; ii++)
  {
    iter=0; float err=HUGE_VAL;
    while(iter++<maxiter && err>epsilon){ // mean-shift iteration
      CvMat y_stub;
      CvMat * y = cvGetSubRect(dst, &y_stub, cvRect(0,ii,ndims,1));
      for (int jj = 0; jj < npoints; jj++){
        dist->data.fl[jj]=
            abs(y->data.fl[0]-CV_MAT_ELEM(*dst,float,jj,0))+
            abs(y->data.fl[1]-CV_MAT_ELEM(*dst,float,jj,1));

        // thresholding -- flat kerenl
        if (dist->data.fl[jj]>ksize) {dist->data.fl[jj]=0.0f;} 
      }
      cvTranspose(dist, dist_t);
      cvMatMul(dist_t, src, numerator);
      double denominator = cvSum(dist).val[0];
      // shift to local mean
      err=0;
      for (int jj=0; jj<ndims; jj++){
        newy[jj]=numerator->data.fl[jj]/denominator;
        CV_MAT_ELEM(*dst,float,ii,jj)=newy[jj];
        err+=pow(y->data.fl[jj]-newy[jj],2);
      }
      err=sqrt(err);

#if 0
      IplImage * dispImage =
          cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);
      cvZero(dispImage);
      // for (int i = 0; i < npoints; i++){
      // 	cvCircle(dispImage, cvPoint(CV_MAT_ELEM(*src,float,i,0),
      //                               CV_MAT_ELEM(*src,float,i,1)),
      //            1,cvScalar(255),-1);
      // }
      for (int i = 0; i < npoints; i++)
        cvCircle(dispImage, cvPoint(CV_MAT_ELEM(*dst,float,i,0),
                                    CV_MAT_ELEM(*dst,float,i,1)),
                 1,cvScalar(0,255),-1);
      cvCircle(dispImage, cvPoint(newy[0],newy[1]),2,cvScalar(0,0,255),-1);
      cvShowImage("Test", dispImage); cvWaitKey();
      cvReleaseImage(&dispImage);
#endif
    } // end of while loop
  } // end of for loop

  delete [] newy;
  cvReleaseMat(&dist);
  cvReleaseMat(&dist_t);
  cvReleaseMat(&numerator);

  __END__;
}
