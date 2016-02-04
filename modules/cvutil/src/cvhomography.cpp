#include "cvhomography.h"

// finds perspective transformation H between
// the object plane and image plane,
// so that (sxi,syi,s) ~ H*(Xi,Yi,1)
void icvFindHomography( const CvMat* object_points,
                        const CvMat* image_points, CvMat* __H )
{
  CvMat *_m = 0, *_M = 0;
  CvMat *_L2 = 0;
    
  CV_FUNCNAME( "cvFindHomography" );

  __BEGIN__;

  int h_type;
  int i, k, count, count2;
  CvPoint2D64f *m, *M;
  CvPoint2D64f cm = {0,0}, sm = {0,0};
  double inv_Hnorm[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
  double H[9];
  CvMat _inv_Hnorm = cvMat( 3, 3, CV_64FC1, inv_Hnorm );
  CvMat _H = cvMat( 3, 3, CV_64FC1, H );
  double LtL[9*9], LW[9], LV[9*9];
  CvMat* _Lp;
  double* L;
  CvMat _LtL = cvMat( 9, 9, CV_64FC1, LtL );
  CvMat _LW = cvMat( 9, 1, CV_64FC1, LW );
  CvMat _LV = cvMat( 9, 9, CV_64FC1, LV );
  CvMat _Hrem = cvMat( 3, 3, CV_64FC1, LV + 8*9 );

  if( !CV_IS_MAT(image_points) || !CV_IS_MAT(object_points) ||
      !CV_IS_MAT(__H) )
    CV_ERROR( CV_StsBadArg, "one of arguments is not a valid matrix" );

  h_type = CV_MAT_TYPE(__H->type);
  if( h_type != CV_32FC1 && h_type != CV_64FC1 )
    CV_ERROR( CV_StsUnsupportedFormat,
              "Homography matrix must have 32fC1 or 64fC1 type" );
  if( __H->rows != 3 || __H->cols != 3 )
    CV_ERROR( CV_StsBadSize, "Homography matrix must be 3x3" );

  count = MAX(image_points->cols, image_points->rows);
  count2 = MAX(object_points->cols, object_points->rows);
  if( count != count2 )
    CV_ERROR( CV_StsUnmatchedSizes,
              "Numbers of image and object points do not match" );

  CV_CALL( _m = cvCreateMat( 1, count, CV_64FC2 ));
  CV_CALL( icvConvertPointsHomogenious( image_points, _m ));
  m = (CvPoint2D64f*)_m->data.ptr;
    
  CV_CALL( _M = cvCreateMat( 1, count, CV_64FC2 ));
  CV_CALL( icvConvertPointsHomogenious( object_points, _M ));
  M = (CvPoint2D64f*)_M->data.ptr;

  // calculate the normalization transformation Hnorm.
  for( i = 0; i < count; i++ )
    cm.x += m[i].x, cm.y += m[i].y;
   
  cm.x /= count; cm.y /= count;

  for( i = 0; i < count; i++ )
  {
    double x = m[i].x - cm.x;
    double y = m[i].y - cm.y;
    sm.x += fabs(x); sm.y += fabs(y);
  }

  sm.x /= count; sm.y /= count;
  inv_Hnorm[0] = sm.x;
  inv_Hnorm[4] = sm.y;
  inv_Hnorm[2] = cm.x;
  inv_Hnorm[5] = cm.y;
  sm.x = 1./sm.x;
  sm.y = 1./sm.y;
    
  CV_CALL( _Lp = _L2 = cvCreateMat( 2*count, 9, CV_64FC1 ) );
  L = _L2->data.db;

  for( i = 0; i < count; i++, L += 18 )
  {
    double x = -(m[i].x - cm.x)*sm.x, y = -(m[i].y - cm.y)*sm.y;
    L[0] = L[9 + 3] = M[i].x;
    L[1] = L[9 + 4] = M[i].y;
    L[2] = L[9 + 5] = 1;
    L[9 + 0] = L[9 + 1] = L[9 + 2] = L[3] = L[4] = L[5] = 0;
    L[6] = x*M[i].x;
    L[7] = x*M[i].y;
    L[8] = x;
    L[9 + 6] = y*M[i].x;
    L[9 + 7] = y*M[i].y;
    L[9 + 8] = y;
  }

  if( count > 4 )
  {
    cvMulTransposed( _L2, &_LtL, 1 );
    _Lp = &_LtL;
  }

  _LW.rows = MIN(count*2, 9);
  cvSVD( _Lp, &_LW, 0, &_LV, CV_SVD_MODIFY_A + CV_SVD_V_T );
  cvScale( &_Hrem, &_Hrem, 1./_Hrem.data.db[8] );
  cvMatMul( &_inv_Hnorm, &_Hrem, &_H );

  if( count > 4 )
  {
    // reuse the available storage for jacobian and other vars
    CvMat _J = cvMat( 2*count, 8, CV_64FC1, _L2->data.db );
    CvMat _err = cvMat( 2*count, 1, CV_64FC1, _L2->data.db + 2*count*8 );
    CvMat _JtJ = cvMat( 8, 8, CV_64FC1, LtL );
    CvMat _JtErr = cvMat( 8, 1, CV_64FC1, LtL + 8*8 );
    CvMat _JtJW = cvMat( 8, 1, CV_64FC1, LW );
    CvMat _JtJV = cvMat( 8, 8, CV_64FC1, LV );
    CvMat _Hinnov = cvMat( 8, 1, CV_64FC1, LV + 8*8 );

    for( k = 0; k < 10; k++ )
    {
      double* J = _J.data.db, *err = _err.data.db;
      for( i = 0; i < count; i++, J += 16, err += 2 )
      {
        double di = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double _xi = (H[0]*M[i].x + H[1]*M[i].y + H[2])*di;
        double _yi = (H[3]*M[i].x + H[4]*M[i].y + H[5])*di;
        err[0] = m[i].x - _xi;
        err[1] = m[i].y - _yi;
        J[0] = M[i].x*di;
        J[1] = M[i].y*di;
        J[2] = di;
        J[8+3] = M[i].x;
        J[8+4] = M[i].y;
        J[8+5] = di;
        J[6] = -J[0]*_xi;
        J[7] = -J[1]*_xi;
        J[8+6] = -J[8+3]*_yi;
        J[8+7] = -J[8+4]*_yi;
        J[3] = J[4] = J[5] = J[8+0] = J[8+1] = J[8+2] = 0.;
      }

      icvGaussNewton( &_J, &_err, &_Hinnov, &_JtJ, &_JtErr, &_JtJW, &_JtJV );

      for( i = 0; i < 8; i++ )
        H[i] += _Hinnov.data.db[i];
    }
  }

  cvConvert( &_H, __H );

  __END__;

  cvReleaseMat( &_m );
  cvReleaseMat( &_M );
  cvReleaseMat( &_L2 );
}

void icvGaussNewton( const CvMat* J, const CvMat* err, CvMat* delta,
                     CvMat* JtJ, CvMat* JtErr, CvMat* JtJW, CvMat* JtJV )
{
  CvMat* _temp_JtJ = 0;
  CvMat* _temp_JtErr = 0;
  CvMat* _temp_JtJW = 0;
  CvMat* _temp_JtJV = 0;
    
  CV_FUNCNAME( "icvGaussNewton" );

  __BEGIN__;

  if( !CV_IS_MAT(J) || !CV_IS_MAT(err) || !CV_IS_MAT(delta) )
    CV_ERROR( CV_StsBadArg,
              "Some of required arguments is not a valid matrix" );

  if( !JtJ )
  {
    CV_CALL( _temp_JtJ = cvCreateMat( J->cols, J->cols, J->type ));
    JtJ = _temp_JtJ;
  }
  else if( !CV_IS_MAT(JtJ) )
    CV_ERROR( CV_StsBadArg, "JtJ is not a valid matrix" );

  if( !JtErr )
  {
    CV_CALL( _temp_JtErr = cvCreateMat( J->cols, 1, J->type ));
    JtErr = _temp_JtErr;
  }
  else if( !CV_IS_MAT(JtErr) )
    CV_ERROR( CV_StsBadArg, "JtErr is not a valid matrix" );

  if( !JtJW )
  {
    CV_CALL( _temp_JtJW = cvCreateMat( J->cols, 1, J->type ));
    JtJW = _temp_JtJW;
  }
  else if( !CV_IS_MAT(JtJW) )
    CV_ERROR( CV_StsBadArg, "JtJW is not a valid matrix" );

  if( !JtJV )
  {
    CV_CALL( _temp_JtJV = cvCreateMat( J->cols, J->cols, J->type ));
    JtJV = _temp_JtJV;
  }
  else if( !CV_IS_MAT(JtJV) )
    CV_ERROR( CV_StsBadArg, "JtJV is not a valid matrix" );

  cvMulTransposed( J, JtJ, 1 );
  cvGEMM( J, err, 1, 0, 0, JtErr, CV_GEMM_A_T );
  cvSVD( JtJ, JtJW, 0, JtJV, CV_SVD_MODIFY_A + CV_SVD_V_T );
  cvSVBkSb( JtJW, JtJV, JtJV, JtErr, delta, CV_SVD_U_T + CV_SVD_V_T );

  __END__;

  if( _temp_JtJ || _temp_JtErr || _temp_JtJW || _temp_JtJV )
  {
    cvReleaseMat( &_temp_JtJ );
    cvReleaseMat( &_temp_JtErr );
    cvReleaseMat( &_temp_JtJW );
    cvReleaseMat( &_temp_JtJV );
  }
}

void icvConvertPointsHomogenious( const CvMat* src, CvMat* dst )
{
  CvMat* temp = 0;
  CvMat* denom = 0;

  CV_FUNCNAME( "cvConvertPointsHomogenious" );

  __BEGIN__;

  int i, s_count, s_dims, d_count, d_dims;
  CvMat _src, _dst, _ones;
  CvMat* ones = 0;

  if( !CV_IS_MAT(src) )
    CV_ERROR( !src ? CV_StsNullPtr : CV_StsBadArg,
              "The input parameter is not a valid matrix" );

  if( !CV_IS_MAT(dst) )
    CV_ERROR( !dst ? CV_StsNullPtr : CV_StsBadArg,
              "The output parameter is not a valid matrix" );

  if( src == dst || src->data.ptr == dst->data.ptr )
  {
    if( src != dst && (!CV_ARE_TYPES_EQ(src, dst) ||
                       !CV_ARE_SIZES_EQ(src,dst)) )
      CV_ERROR( CV_StsBadArg, "Invalid inplace operation" );
    EXIT;
  }

  if( src->rows > src->cols )
  {
    if( !((src->cols > 1) ^ (CV_MAT_CN(src->type) > 1)) )
      CV_ERROR( CV_StsBadSize,
                "Either the number of channels or columns or "
                "rows must be =1" );

    s_dims = CV_MAT_CN(src->type)*src->cols;
    s_count = src->rows;
  }
  else
  {
    if( !((src->rows > 1) ^ (CV_MAT_CN(src->type) > 1)) )
      CV_ERROR( CV_StsBadSize,
                "Either the number of channels or columns or "
                "rows must be =1" );

    s_dims = CV_MAT_CN(src->type)*src->rows;
    s_count = src->cols;
  }

  if( src->rows == 1 || src->cols == 1 )
    src = cvReshape( src, &_src, 1, s_count );

  if( dst->rows > dst->cols )
  {
    if( !((dst->cols > 1) ^ (CV_MAT_CN(dst->type) > 1)) )
      CV_ERROR( CV_StsBadSize,
                "Either the number of channels or columns or "
                "rows in the input matrix must be =1" );

    d_dims = CV_MAT_CN(dst->type)*dst->cols;
    d_count = dst->rows;
  }
  else
  {
    if( !((dst->rows > 1) ^ (CV_MAT_CN(dst->type) > 1)) )
      CV_ERROR( CV_StsBadSize,
                "Either the number of channels or columns or "
                "rows in the output matrix must be =1" );

    d_dims = CV_MAT_CN(dst->type)*dst->rows;
    d_count = dst->cols;
  }

  if( dst->rows == 1 || dst->cols == 1 )
    dst = cvReshape( dst, &_dst, 1, d_count );

  if( s_count != d_count )
    CV_ERROR( CV_StsUnmatchedSizes, "Both matrices must have the "
                "same number of points" );

  if( CV_MAT_DEPTH(src->type) < CV_32F || CV_MAT_DEPTH(dst->type) < CV_32F )
    CV_ERROR( CV_StsUnsupportedFormat,
              "Both matrices must be floating-point "
                "(single or double precision)" );

  if( s_dims < 2 || s_dims > 4 || d_dims < 2 || d_dims > 4 )
    CV_ERROR( CV_StsOutOfRange,
              "Both input and output point dimensionality "
                "must be 2, 3 or 4" );

  if( s_dims < d_dims - 1 || s_dims > d_dims + 1 )
    CV_ERROR( CV_StsUnmatchedSizes,
              "The dimensionalities of input and output "
                "point sets differ too much" );

  if( s_dims == d_dims - 1 )
  {
    if( d_count == dst->rows )
    {
      ones = cvGetSubRect( dst, &_ones, cvRect( s_dims, 0, 1, d_count ));
      dst = cvGetSubRect( dst, &_dst, cvRect( 0, 0, s_dims, d_count ));
    }
    else
    {
      ones = cvGetSubRect( dst, &_ones, cvRect( 0, s_dims, d_count, 1 ));
      dst = cvGetSubRect( dst, &_dst, cvRect( 0, 0, d_count, s_dims ));
    }
  }

  if( s_dims <= d_dims )
  {
    if( src->rows == dst->rows && src->cols == dst->cols )
    {
      if( CV_ARE_TYPES_EQ( src, dst ) )
        cvCopy( src, dst );
      else
        cvConvert( src, dst );
    }
    else
    {
      if( !CV_ARE_TYPES_EQ( src, dst ))
      {
        CV_CALL( temp = cvCreateMat( src->rows, src->cols, dst->type ));
        cvConvert( src, temp );
        src = temp;
      }
      cvTranspose( src, dst );
    }

    if( ones )
      cvSet( ones, cvRealScalar(1.) );
  }
  else
  {
    int s_plane_stride, s_stride, d_plane_stride, d_stride, elem_size;

    if( !CV_ARE_TYPES_EQ( src, dst ))
    {
      CV_CALL( temp = cvCreateMat( src->rows, src->cols, dst->type ));
      cvConvert( src, temp );
      src = temp;
    }

    elem_size = CV_ELEM_SIZE(src->type);

    if( s_count == src->cols )
      s_plane_stride = src->step / elem_size, s_stride = 1;
    else
      s_stride = src->step / elem_size, s_plane_stride = 1;

    if( d_count == dst->cols )
      d_plane_stride = dst->step / elem_size, d_stride = 1;
    else
      d_stride = dst->step / elem_size, d_plane_stride = 1;

    CV_CALL( denom = cvCreateMat( 1, d_count, dst->type ));

    if( CV_MAT_DEPTH(dst->type) == CV_32F )
    {
      const float* xs = src->data.fl;
      const float* ys = xs + s_plane_stride;
      const float* zs = 0;
      const float* ws = xs + (s_dims - 1)*s_plane_stride;

      float* iw = denom->data.fl;

      float* xd = dst->data.fl;
      float* yd = xd + d_plane_stride;
      float* zd = 0;

      if( d_dims == 3 )
      {
        zs = ys + s_plane_stride;
        zd = yd + d_plane_stride;
      }

      for( i = 0; i < d_count; i++, ws += s_stride )
      {
        float t = *ws;
        iw[i] = t ? t : 1.f;
      }

      cvDiv( 0, denom, denom );

      if( d_dims == 3 )
        for( i = 0; i < d_count; i++ )
        {
          float w = iw[i];
          float x = *xs * w, y = *ys * w, z = *zs * w;
          xs += s_stride; ys += s_stride; zs += s_stride;
          *xd = x; *yd = y; *zd = z;
          xd += d_stride; yd += d_stride; zd += d_stride;
        }
      else
        for( i = 0; i < d_count; i++ )
        {
          float w = iw[i];
          float x = *xs * w, y = *ys * w;
          xs += s_stride; ys += s_stride;
          *xd = x; *yd = y;
          xd += d_stride; yd += d_stride;
        }
    }
    else
    {
      const double* xs = src->data.db;
      const double* ys = xs + s_plane_stride;
      const double* zs = 0;
      const double* ws = xs + (s_dims - 1)*s_plane_stride;

      double* iw = denom->data.db;

      double* xd = dst->data.db;
      double* yd = xd + d_plane_stride;
      double* zd = 0;

      if( d_dims == 3 )
      {
        zs = ys + s_plane_stride;
        zd = yd + d_plane_stride;
      }

      for( i = 0; i < d_count; i++, ws += s_stride )
      {
        double t = *ws;
        iw[i] = t ? t : 1.;
      }

      cvDiv( 0, denom, denom );

      if( d_dims == 3 )
        for( i = 0; i < d_count; i++ )
        {
          double w = iw[i];
          double x = *xs * w, y = *ys * w, z = *zs * w;
          xs += s_stride; ys += s_stride; zs += s_stride;
          *xd = x; *yd = y; *zd = z;
          xd += d_stride; yd += d_stride; zd += d_stride;
        }
      else
        for( i = 0; i < d_count; i++ )
        {
          double w = iw[i];
          double x = *xs * w, y = *ys * w;
          xs += s_stride; ys += s_stride;
          *xd = x; *yd = y;
          xd += d_stride; yd += d_stride;
        }
    }
  }

  __END__;

  cvReleaseMat( &denom );
  cvReleaseMat( &temp );
}
