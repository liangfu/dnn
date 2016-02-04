/**
 * @file   cvparticleutil.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Feb 28 17:29:59 2014
 * 
 * @brief  
 * 
 * 
 */
#include "cvparticleutil.h"

CVAPI(CvScalar) cvAngleMean( const CvArr * arr, 
                             const CvArr * weight , 
                             double wrap  )
{
  CvMat* mat, matstub;
  CvMat* wmat = NULL, wmatstub;
  CvScalar mean = cvScalar(0,0,0,0);
  int row, col, ch;
  int nChannels;
  CvScalar elem;
  CvScalar mean_cos = cvScalar(0,0,0,0);
  CvScalar mean_sin = cvScalar(0,0,0,0);
  CvScalar welem;
  CV_FUNCNAME( "cvAngleMean" );
  __BEGIN__;

  mat = (CvMat*)arr;
  if( !CV_IS_MAT(mat) )
  {
	CV_CALL( mat = cvGetMat( mat, &matstub ) );
  }
  if( weight != NULL )
  {
	wmat = (CvMat*)weight;
	if( !CV_IS_MAT(wmat) )
	{
	  CV_CALL( wmat = cvGetMat( wmat, &wmatstub ) );
	}
	CV_ASSERT( 
			  mat->rows == wmat->rows && 
			  mat->cols == wmat->cols &&
			  CV_MAT_CN(mat->type) == CV_MAT_CN(wmat->type) 
			   );
  }
  nChannels = CV_MAT_CN(mat->type);
  if( wmat == NULL ) // uniform
  {
	double w = 1.0 / (double)mat->rows * (double)mat->cols;
	welem = cvScalar( w, w, w, w );
  }
  for( row = 0; row < mat->rows; row++ )
  {
	for( col = 0; col < mat->cols; col++ )
	{
	  elem = cvGet2D( mat, row, col );
	  if( wmat != NULL ) welem = cvGet2D( wmat, row, col );
	  for( ch = 0; ch < nChannels; ch++ )
	  {
		mean_cos.val[ch] += 
		  cos( elem.val[ch] * 2*M_PI / wrap ) * welem.val[ch];
		mean_sin.val[ch] += 
		  sin( elem.val[ch] * 2*M_PI / wrap ) * welem.val[ch];
	  }
	}
  }
  for( ch = 0; ch < nChannels; ch++ )
  {
	mean.val[ch] = 
	  atan( mean_sin.val[ch] / mean_cos.val[ch] ) * wrap / (2*M_PI);
  }
  __END__;
  return mean;
}

CVAPI(CvScalar) cvLogSum( const CvArr *arr )
{
  IplImage* img = (IplImage*)arr, imgstub;
  IplImage *tmp, *tmp2;
  int ch;
  CvScalar sumval;
  CvScalar minval, maxval;
  CV_FUNCNAME( "cvLogSum" );
  __BEGIN__;

  if( !CV_IS_IMAGE(img) )
  {
	CV_CALL( img = cvGetImage( img, &imgstub ) );
  }
  tmp = cvCreateImage( cvGetSize(img), img->depth, img->nChannels );
  tmp2 = cvCreateImage( cvGetSize(img), img->depth, img->nChannels );

  // to avoid loss of precision caused by taking exp as much as possible
  // if this trick is not required, cvExp -> cvSum are enough
  for( ch = 0; ch < img->nChannels; ch++ )
  {
	cvSetImageCOI( img, ch + 1 );
	cvMinMaxLoc( img, &minval.val[ch], &maxval.val[ch] );
  }
  cvSetImageCOI( img, 0 );
  cvSubS( img, maxval, tmp );

  cvExp( tmp, tmp2 );
  sumval = cvSum( tmp2 );
  for( ch = 0; ch < img->nChannels; ch++ )
  {
	sumval.val[ch] = log( sumval.val[ch] ) + maxval.val[ch];
  }
  cvReleaseImage( &tmp );
  cvReleaseImage( &tmp2 );
  __END__;
  return sumval;
}

CVAPI(void) cvSetCols( const CvArr* src, CvArr* dst,
                       int start_col, int end_col )
{
  int coi;
  CvMat *srcmat = (CvMat*)src, srcmatstub;
  CvMat *dstmat = (CvMat*)dst, dstmatstub;
  CvMat *refmat, refmathdr;
  int cols;

  CV_FUNCNAME( "cvSetCols" );
  __BEGIN__;

  if( !CV_IS_MAT(dstmat) )
  {
    CV_CALL( dstmat = cvGetMat( dstmat, &dstmatstub, &coi ) );
    if (coi != 0) CV_ERROR_FROM_CODE(CV_BadCOI);
  }
  if( !CV_IS_MAT(srcmat) )
  {
    CV_CALL( srcmat = cvGetMat( srcmat, &srcmatstub, &coi ) );
    if (coi != 0) CV_ERROR_FROM_CODE(CV_BadCOI);
  }

  cols = end_col - start_col;
  CV_ASSERT( srcmat->cols == cols || dstmat->cols == cols );

  if( srcmat->cols == cols )
  {
    refmat = cvGetCols( dstmat, &refmathdr, start_col, end_col );
    cvCopy( srcmat, refmat );
  }
  else
  {
    refmat = cvGetCols( srcmat, &refmathdr, start_col, end_col );
    cvCopy( refmat, dstmat );
  }

  __END__;
}

CVAPI(void) cvSetRows( const CvArr* src, CvArr* dst,
                       int start_row, int end_row,
                       int delta_row  )
{
  int coi;
  CvMat *srcmat = (CvMat*)src, srcmatstub;
  CvMat *dstmat = (CvMat*)dst, dstmatstub;
  CvMat *refmat, refmathdr;
  int rows;
  CV_FUNCNAME( "cvSetRows" );
  __BEGIN__;
  if( !CV_IS_MAT(dstmat) )
  {
	CV_CALL( dstmat = cvGetMat( dstmat, &dstmatstub, &coi ) );
	if (coi != 0) CV_ERROR_FROM_CODE(CV_BadCOI);
  }
  if( !CV_IS_MAT(srcmat) )
  {
	CV_CALL( srcmat = cvGetMat( srcmat, &srcmatstub, &coi ) );
	if (coi != 0) CV_ERROR_FROM_CODE(CV_BadCOI);
  }
  rows = cvFloor( ( end_row - start_row ) / delta_row );
  CV_ASSERT( srcmat->rows == rows || dstmat->rows == rows );
  if( srcmat->rows == rows )
  {
	refmat = cvGetRows( dstmat, &refmathdr, start_row, end_row, delta_row );
	cvCopy( srcmat, refmat );
  }
  else
  {
	refmat = cvGetRows( srcmat, &refmathdr, start_row, end_row, delta_row );
	cvCopy( refmat, dstmat );
  }
  __END__;
}

