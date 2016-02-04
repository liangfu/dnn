/**
 * @file   main22_jet.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Tue Jan  8 10:30:23 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"
#include "lut_jet.h"

double interpolate( double val, double y0, double x0, double y1, double x1 ) {
  return (val-x0)*(y1-y0)/(x1-x0) + y0;
}

double base( double val ) {
  if ( val <= -0.75 ) return 0;
  else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
  else if ( val <= 0.25 ) return 1.0;
  else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
  else return 0.0;
}

double red( double gray ) {
  return base( gray - 0.5 );
}
double green( double gray ) {
  return base( gray );
}
double blue( double gray ) {
  return base( gray + 0.5 );
}

// double interpolate( double val, double y0, double x0, double y1, double x1 ) {
//   return (val-x0)*(y1-y0)/(x1-x0) + y0;
// }

// double blue( double grayscale ) {
//   if ( grayscale < -0.33 )
//     return 1.0;
//   else if ( grayscale < 0.33 )
//     return interpolate( grayscale, 1.0, -0.33, 0.0, 0.33 );
//   else return 0.0;
// }

// double green( double grayscale ) {
//   if ( grayscale < -1.0 ) return 0.0; // unexpected grayscale value
//   if  ( grayscale < -0.33 )
//     return interpolate( grayscale, 0.0, -1.0, 1.0, -0.33 );
//   else if ( grayscale < 0.33 ) return 1.0;
//   else if ( grayscale <= 1.0 )
//     return interpolate( grayscale, 1.0, 0.33, 0.0, 1.0 );
//   else return 1.0; // unexpected grayscale value
// }

// double red( double grayscale ) {
//   if ( grayscale < -0.33 ) return 0.0;
//   else if ( grayscale < 0.33 )
//     return interpolate( grayscale, 0.0, -0.33, 1.0, 0.33 );
//   else return 1.0;
// }

void cvColorMapLUT(CvArr * rawImage, CvArr * dispImage, CvArr * lut);


int main(int argc, char * argv[])
{
  CvMat * rawImage = cvCreateMat(256, 24, CV_8U);
  CvMat * dispImage = cvCreateMat(256, 24, CV_8UC3);
  CvMat * tmpImage = cvCreateMat(256, 24, CV_32F);
  int i;

  // construct raw image
  for ( i = 0; i < rawImage->rows; i++ )
  {
    CvMat hdr;
    CvMat * row = cvGetRow(rawImage, &hdr, i);
    cvSet(row, cvScalar(i));
  }
#if 0
  cvScale(rawImage, tmpImage, 1./255.*2.);
  cvScale(tmpImage, tmpImage, 1., -1.);

  for ( i = 0; i < rawImage->rows; i++ )
  {
    double val = CV_MAT_ELEM(*tmpImage, float, i, 0);
    CvMat hdr;
    CvMat * row = cvGetRow(dispImage, &hdr, i);
    CvScalar color =
        cvScalar(blue(val)*255., green(val)*255., red(val)*255.);
    cvSet(row, color);
    fprintf(stderr, "/* %3d */ %d, %d, %d,\n",
            i,     // intensity
            cvRound(color.val[0]), // blue
            cvRound(color.val[1]), // green
            cvRound(color.val[2])  // red
            );
  }
#else
  CvMat jetImage = cvMat(256, 3, CV_8U, xrcLUT_JET);
  // memcpy(jetImage->data.ptr, xrcLUT_JET, sizeof(256*3*sizeof(uchar)));
  CV_TIMER_START();
  cvColorMapLUT(rawImage, dispImage, &jetImage);
  CV_TIMER_SHOW();
  cvShowImage("Test", &jetImage); CV_WAIT();
  // cvReleaseMat(&jetImage);
#endif
  cvShowImage("Test", dispImage); CV_WAIT();

  cvReleaseMat(&rawImage);
  cvReleaseMat(&dispImage);
  cvReleaseMat(&tmpImage);
  
  return 0;
}

void cvColorMapLUT(CvArr * _src, CvArr * _dst, CvArr * _lut)
{
  CvMat * src, src_stub, * dst, dst_stub, * lut, lut_stub;
  int i, j;
  
  CV_FUNCNAME("cvColorMapLUT");
  __BEGIN__;

  if (!CV_IS_MAT(_src)) {
    src = cvGetMat(_src, &src_stub);
  }else{
    src = (CvMat*)_src;
  }
  
  if (!CV_IS_MAT(_dst)) {
    dst = cvGetMat(_dst, &dst_stub);
  }else{
    dst = (CvMat*)_dst;
  }

  if (!CV_IS_MAT(_lut)) {
    lut = cvGetMat(_lut, &lut_stub);
  }else{
    lut = (CvMat*)_lut;
  }

  CV_ASSERT(CV_MAT_TYPE(src->type)==CV_8U);
  CV_ASSERT(CV_MAT_TYPE(dst->type)==CV_8UC3);
  
  typedef struct { uchar val[3]; } color_t;
  for (i = 0; i < src->rows; i++)
    for (j = 0; j < src->cols; j++)
    {
      color_t color;
      uchar intensity = CV_MAT_ELEM(*src, uchar, i, j);
      color.val[0] = CV_MAT_ELEM(*lut, uchar, intensity, 0);
      color.val[1] = CV_MAT_ELEM(*lut, uchar, intensity, 1);
      color.val[2] = CV_MAT_ELEM(*lut, uchar, intensity, 2);
      CV_MAT_ELEM(*dst, color_t, i, j) = color;
    }
  
  __END__;
}

/* template
void cvColorMapLUT(CvArr * _src, CvArr * _dst, CvArr * _lut)
{
  CvMat * src, src_stub, * dst, dst_stub, * lut, lut_stub;
  
  CV_FUNCNAME("cvColorMapLUT");
  __BEGIN__;

  if (!CV_IS_MAT(_src)) {
    src = cvGetMat(_src, &src_stub);
  }else{
    src = (CvMat*)_src;
  }
  
  if (!CV_IS_MAT(_dst)) {
    dst = cvGetMat(_dst, &dst_stub);
  }else{
    dst = (CvMat*)_dst;
  }

  if (!CV_IS_MAT(_lut)) {
    lut = cvGetMat(_lut, &lut_stub);
  }else{
    lut = (CvMat*)_lut;
  }

  
  
  __END__;
}
*/
