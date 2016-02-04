/**
 * @file   cvparticleutil.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Feb 28 17:26:28 2014
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_PARTICLE_UTIL_H__
#define __CV_PARTICLE_UTIL_H__

#include "cvext_c.h"

#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>

/**
 * Compute mean of angle elements of an array (each channel independently). 
 *
 * There is a fact that 0 degrees and 360 degrees are identical angles, 
 * so that for example 180 degrees is not a sensible mean of 2 degrees and 
 * 358 degrees, but 0 degree is the mean. 
 * Algorithm works as 1. compute means of cosine and sine 
 * 2. take arc tangent of the mean consine and sine. 
 *
 * @param  arr     array
 * @param  weight  Weight to compute mean. The deafult is 1/num (uniform).
 *                 The size must be same with arr. 
 * @param  wrap    The unit of wrapping around.
 *                 The defeault is 360 as angle.
 * @return angle mean for each channel
 */
CVAPI(CvScalar) cvAngleMean( const CvArr *arr, 
                             const CvArr *weight CV_DEFAULT(NULL), 
                             double wrap CV_DEFAULT(360) );

/**
 * Set array col or col span
 *
 * Following code is faster than using this function because it does not 
 * require cvCopy()
 * @code
 * CvMat* submat, submathdr;
 * submat = cvGetCols( mat, &submathdr, start_col, end_col, delta_col );
 * // Write on submat
 * @endcode
 *
 * @param src       Source array
 * @param dst       Target array. Either of array must be size of setting cols.
 * @param start_col Zero-based index of the starting col (inclusive) of the span. 
 * @param end_col   Zero-based index of the ending col (exclusive) of the span. 
 * @return CVAPI(void)
 * @see cvSetCol( src, dst, col ) // cvSetCols( src, dst, col, col + 1 )
 */
CVAPI(void) cvSetCols( const CvArr* src, CvArr* dst,
                       int start_col, int end_col );

/**
 * Set array col
 *
 * #define cvSetCol(src, dst, col) (cvSetCols( src, dst, col, col + 1))
 */
CV_INLINE void cvSetCol( const CvArr* src, CvArr* dst, int col )
{
    cvSetCols( src, dst, col, col+1 );
}

/**
 * Set array row or row span
 *
 * Following code is faster than using this function because it does not 
 * require cvCopy()
 * @code
 * CvMat* submat, submathdr;
 * submat = cvGetRows( mat, &submathdr, start_row, end_row, delta_row );
 * // Write on submat
 * @endcode
 *
 * @param src       Source array
 * @param dst       Target array. Either of array must be size of setting rows.
 * @param start_row Zero-based index of the starting row (inclusive) of the span. 
 * @param end_row   Zero-based index of the ending row (exclusive) of the span. 
 * @param [delta_row = 1]
 *                  Index step in the row span. That is, the function extracts every 
 *                  delta_row-th row from start_row and up to (but not including) end_row. 
 * @return CVAPI(void)
 * @see cvSetRow( src, dst, row ) // cvSetRows( src, dst, row, row + 1 )
 */

CVAPI(void) cvSetRows( const CvArr* src, CvArr* dst,
                       int start_row, int end_row,
                       int delta_row CV_DEFAULT(1) );

/**
 * Set array row
 *
 * #define cvSetRow(src, dst, row) (cvSetRows( src, dst, row, row + 1))
 */
CV_INLINE void cvSetRow( const CvArr* src, CvArr* dst, int row )
{
    cvSetRows( src, dst, row, row+1 );
}

/**
 * This function returns a Gaussian random variate, with mean zero and standard deviation sigma.
 *
 * @param rng cvRNG random state
 * @param sigma standard deviation
 * @return double
 */
CV_INLINE double cvRandGauss( CvRNG* rng, double sigma )
{
    CvMat* mat = cvCreateMat( 1, 1, CV_64FC1 );
    double var = 0;
    cvRandArr( rng, mat, CV_RAND_NORMAL, cvRealScalar(0), cvRealScalar(sigma) );
    var = cvmGet( mat, 0, 0 );
    cvReleaseMat( &mat );
    return var;
}

/**
 * Compute log(sum) of log values
 *
 * Get log(a + b + c) from log(a), log(b), log(c)
 * Useful to take sum of probabilities from log probabilities
 * Useful to avoid loss of precision caused by taking exp
 *
 * @param  arr       array having log values. 32F or 64F
 * @return CvScalar  log sum for each channel
 */
CVAPI(CvScalar) cvLogSum( const CvArr *arr );

#endif // __CV_PARTICLE_UTIL_H__
