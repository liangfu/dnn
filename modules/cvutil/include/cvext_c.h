/**
 * @file   cvext_c.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Nov 19 11:00:33 2012
 * 
 * @brief  addtional functions for programming using OpenCV
 *
 * Modifications:
 *   date generate by command `date +"%R %F"`
 *
 * ======================  15:20 2012-11-08  ======================
 * * implement CvAbstractTracker class and CvActiveContour class
 * ======================  09:29 2013-08-07  ======================
 * * implement icvTranspose and icvDiag functions for convenient matrix
 *   transforms.
 */

#ifndef __CV_EXT_C_H__
#define __CV_EXT_C_H__

#include <stdio.h>
#include <stdlib.h>

#include <cxcore.h>
#include <cv.h>
#ifndef ANDROID
#include <highgui.h>
#else
#include <android/log.h>
#endif

//-------------------------------------------------------
// GLOBAL DEFINITIONS AND ENUMS
//-------------------------------------------------------

#define CV_MAXSTRLEN 1024

#define CV_RED   cvScalar(0,0,255)
#define CV_GREEN cvScalar(0,255,0)
#define CV_BLUE  cvScalar(255,0,0)
#define CV_BLACK cvScalarAll(0)
#define CV_WHITE cvScalarAll(255)

#define CV_BGR2YUV CV_BGR2YCrCb

enum //CvColorMapType
{
  CV_CM_AUTUMN = 0,
  CV_CM_BONE = 1,
  CV_CM_GRAY = 1,
  CV_CM_JET = 2,
  CV_CM_WINTER = 3,
  CV_CM_RAINBOW = 4,
  CV_CM_OCEAN = 5,
  CV_CM_SUMMER = 6,
  CV_CM_SPRING = 7,
  CV_CM_COOL = 8,
  CV_CM_HSV = 9,
  CV_CM_PINK = 10,
  CV_CM_HOT = 11
};

CV_INLINE void cvDoNothing() { }

// redefine a list of macros for compatible with OpenCV-2.x series
#if CV_MAJOR_VERSION==2
// #define __BEGIN__ __CV_BEGIN__
// #define __END__ __CV_END__
#ifndef __BEGIN__
#define __BEGIN__
#endif //__BEGIN__

#ifndef __END__
#define __END__
#endif //__END__

#ifndef EXIT
#define EXIT 
#endif // EXIT

// #ifdef CV_ASSERT
// #undef CV_ASSERT
// #endif // CV_ASSERT
// #define CV_ASSERT assert

#ifdef CV_CALL
#undef CV_CALL
#endif // CV_CALL
#define CV_CALL OPENCV_CALL

#ifdef CV_ERROR
#undef CV_ERROR
#endif // CV_ERROR
#define CV_ERROR(code,msg) OPENCV_ERROR(code, cvFuncName, (msg))

#ifdef CV_ERROR_FROM_CODE
#undef CV_ERROR_FROM_CODE
#endif // CV_ERROR_FROM_CODE
#define CV_ERROR_FROM_CODE(code) OPENCV_ERROR(code, cvFuncName, "")

#endif // CV_MAJOR_VERSION==2

#if defined(WIN32) || defined(_WIN32)
#include <windows.h>
#ifndef __func__
#define __func__ __FUNCTION__
#endif
#define LOGE(fmt,...)                                               \
  do {                                                              \
    char msgbuf[1024];sprintf(msgbuf,"ERROR: " fmt,##__VA_ARGS__);  \
    MessageBoxA(NULL,msgbuf,"ERROR",MB_ICONERROR|MB_OK);            \
  }while(0)
#define LOGW(fmt,...)                                                   \
  do {                                                                  \
    char msgbuf[1024];sprintf(msgbuf,"WARNING: " fmt,##__VA_ARGS__);    \
    MessageBoxA(NULL,msgbuf,"WARNING",MB_ICONWARNING|MB_OK);            \
  }while(0)
#define LOGI(fmt,...)                                               \
  do {                                                              \
    char msgbuf[1024];sprintf(msgbuf,"INFO: " fmt,##__VA_ARGS__);   \
    MessageBoxA(NULL,msgbuf,"INFO",MB_ICONINFORMATION|MB_OK);       \
  }while(0)
#elif defined(ANDROID)
#define LOG_TAG "MobiFRsp"
#define LOGE(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,      \
                        "ERROR: " fmt,##__VA_ARGS__);   \
  }while(0)
#define LOGW(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_WARN,LOG_TAG,       \
                        "WARNING: " fmt,##__VA_ARGS__); \
  }while(0)
#define LOGI(fmt,...)                                   \
  do {                                                  \
    __android_log_print(ANDROID_LOG_INFO,LOG_TAG,       \
                        "INFO: " fmt,##__VA_ARGS__);    \
  }while(0)
#elif defined(__linux__)
#define LOGE(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"ERROR: " fmt "\n",##__VA_ARGS__);   \
  }while(0)
#define LOGW(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"WARNING: " fmt "\n",##__VA_ARGS__); \
  }while(0)
#define LOGI(fmt,...)                                   \
  do {                                                  \
    fprintf(stderr,"INFO: " fmt "\n",##__VA_ARGS__);    \
  }while(0)
#else 
#endif

//-------------------------------------------------------
// MEMORY MANAGEMENT FUNCTIONS
//-------------------------------------------------------

CV_INLINE
int icvIsBigEndian()
{
  union{unsigned long a;unsigned char b[4];}u;
  u.a=1;
  return u.b[3];
}

CV_INLINE
int icvFileExist(const char * file)
{
  int retval=0;
  FILE * fp = fopen(file,"r");
  if (fp){retval=1;fclose(fp);}
  else {retval=0;}
  return retval;
}

CVAPI(void) icvPrintCompilerInfo();

/** 
 * Release IplImage/CvMat with NULL reference check
 * 
 * @param im    IN/OUT: image/matrix to be released
 */
CV_INLINE
void cvReleaseImageEx(IplImage *& im)
{
  if ( im ){ cvReleaseImage(&im);im = NULL; }
}

CV_INLINE
void cvReleaseMatEx(CvMat *& mat)
{
  if ( mat ){ cvReleaseMat(&mat); mat = NULL; }
}

//-------------------------------------------------------
// TYPE CONVERSION FUNCTIONS
//-------------------------------------------------------

typedef struct CvComplex { float real; float imag; } CvComplex;

#if 0
/**
 * Describe rectangle region with floating type parameters
 * 
 */
typedef struct _CvRect32f {
  float x;
  float y;
  float width;
  float height;
} CvRect32f;

/**
 * Contructor of CvRect32f
 * 
 * @param x         IN: left corner of the rectangle
 * @param y         IN: top corner of rectangle
 * @param width     IN: width of the rectangle
 * @param height    IN: height of the rectangle
 * 
 * @return          OUT: 
 */
CV_INLINE
CvRect32f cvRect32f(float x, float y, float width, float height)
{
  CvRect32f rect;
  rect.x = x; rect.y = y;
  rect.width = width;
  rect.height = height;
  return rect;
}

/**
 * convert from CvRect32f to CvRect
 * 
 */
CV_INLINE
CvRect cvRectFrom32f(CvRect32f r32)
{
  CvRect r;
  r.x = cvRound(r32.x);
  r.y = cvRound(r32.y);
  r.width = cvRound(r32.width);
  r.height = cvRound(r32.height);
  return r;
}

/**
 * convert CvRect32f to CvRect
 * 
 */
CV_INLINE
CvRect32f cvRectTo32f(CvRect r)
{
  CvRect32f r32;
  r32.x = r.x; r32.y = r.y;
  r32.width = r.width;
  r32.height = r.height;
  return r32;
}
#endif // declaration of CvRect32f

/** 
 * Transform between CvPoint and CvPoint2D64f
 */
CV_INLINE
CvPoint2D64f cvPointTo64f(CvPoint p)
{
  return cvPoint2D64f(p.x, p.y);
}

CV_INLINE
CvPoint cvPointFrom64f(CvPoint2D64f p)
{
  return cvPoint(cvRound(p.x), cvRound(p.y));
}

/** 
 * compare function for two CvPoint elements:
 * 
 * @param pt1 IN: first element
 * @param pt2 IN: second element
 * 
 * @return OUT: whether they are equal or not 
 */
CV_INLINE
bool operator==(const CvPoint & pt1, const CvPoint & pt2)
{
  return ( (pt1.x==pt2.x)&&(pt1.y==pt2.y) );
}

CV_INLINE
bool operator!=(const CvPoint & pt1, const CvPoint & pt2)
{
  return ( (pt1.x!=pt2.x)||(pt1.y!=pt2.y) );
}

CV_INLINE
bool operator==(const CvRect & r1, const CvRect & r2)
{
  return ( (r1.x==r2.x)&&(r1.y==r2.y)&&
           (r1.width==r2.width)&&(r1.height==r2.height) );
}

CV_INLINE
bool operator!=(const CvRect & r1, const CvRect & r2)
{
  return ( (r1.x!=r2.x)||(r1.y!=r2.y)||
           (r1.width!=r2.width)||(r1.height!=r2.height) );
}

CV_INLINE
bool operator==(const CvSize & s1, const CvSize & s2)
{
  return (s1.width==s2.width)&&(s1.height==s2.height);
}

CV_INLINE
bool operator!=(const CvSize & s1, const CvSize & s2)
{
  return (s1.width!=s2.width)||(s1.height!=s2.height);
}

/** 
 * convert between index and coordinate of a point
 * 
 * @param idx   IN: index of a point
 * @param ncols IN: width of the image
 * 
 * @return OUT: position of the point, which is defined by the index
 */
CV_INLINE
CvPoint cvIdx2Vtx(int idx, int ncols)
{
  return cvPoint(idx%ncols, idx/ncols);
}

CV_INLINE
int cvVtx2Idx(int x, int y, int ncols)
{
  return y*ncols+x;
}

/** 
 * 
 * 
 * @param a 
 * @param b 
 * @param s 
 */
CV_INLINE
void swap(void * a, void * b, size_t sz)
{
  void * tmp = malloc(sz);

  memcpy(tmp,b,sz);
  memcpy(b,a,sz);
  memcpy(a,tmp,sz);

  free(tmp);
}

/** 
 * return sign of given input
 * 
 * @param val IN: 
 * 
 * @return OUT: return 1 if input is larger than 0, return -1 otherwise.
 */
CV_INLINE
int cvSign(int    val) {
  return (val==0)?0:(val>0?1:-1);}
CV_INLINE
int cvSign(float  val) {
  return ((val<1e-5)&&(val>-1e-5))?0:(val>0?1:-1);}
CV_INLINE
int cvSign(double val) {
  return ((val<1e-5)&&(val>-1e-5))?0:(val>0?1:-1);}

// |error| < 0.005
CV_INLINE
float icvFastAtan2( float y, float x )
{
  if ( x == 0.0f )
  {
    if ( y > 0.0f ) return 1.5707963f;
    if ( y == 0.0f ) return 0.0f;
    return -1.5707963f;
  }
  float atan;
  float z = y/x;
  if ( fabsf( z ) < 1.0f )
  {
    atan = z/(1.0f + 0.28f*z*z);
    if ( x < 0.0f )
    {
      if ( y < 0.0f ) return atan - 3.14159265f;
      return atan + 3.14159265f;
    }
  }
  else
  {
    atan = 1.5707963f - z/(z*z + 0.28f);
    if ( y < 0.0f ) return atan - 3.14159265f;
  }
  return atan;
}

CV_INLINE
void icvTranspose(CvMat * src, CvMat * dst)
{
  int i,j,nr=src->rows,nc=src->cols;
  int type=CV_MAT_TYPE(src->type);
  int srcstep,dststep;
  assert((dst->rows==nc)&&(dst->cols==nr));
  assert(CV_MAT_TYPE(dst->type)==type);
  if (type==CV_32F){
  float * fsptr = src->data.fl;
  float * fdptr = dst->data.fl;
  srcstep=src->step/sizeof(float);
  dststep=dst->step/sizeof(float);
  for (i=0;i<nr;i++){
  for (j=0;j<nc;j++){
    (fdptr+j*dststep)[i]=(fsptr+i*srcstep)[j];
  }    
  }
  }else{
    fprintf(stderr,"ERROR: type not supported in 'icvTranpose' function!\n");
    assert(false);
  }
}

/** 
 * convert between diagonal matrix and its vector form
 */
CV_INLINE
void icvDiag(CvMat * src, CvMat * dst)
{
  int vec2mat=1,N,i,step;
  assert(CV_MAT_TYPE(src->type)==CV_32F);
  assert(CV_MAT_TYPE(dst->type)==CV_32F);
  float * srcptr=src->data.fl;
  float * dstptr=dst->data.fl;
  if ((src->cols==1)||(src->rows==1)){vec2mat=1;}
  if (vec2mat){
    cvZero(dst);
    if (src->cols==1){N=src->rows;}else{N=src->cols;}
    assert(N==MIN(dst->cols,dst->rows));
    step=dst->step/sizeof(float);
    for (i=0;i<N;i++){ (dstptr+i*step)[i]=srcptr[i]; }
  }else{
    if (dst->cols==1){N=dst->rows;}else{N=dst->cols;}
    assert(N==MIN(src->cols,src->rows));
    step=src->step/sizeof(float);
    for (i=0;i<N;i++){ dstptr[i]=(srcptr+i*step)[i]; }
  }
}

CV_INLINE
CvPoint cvGetCenter(CvRect roi)
{
  return cvPoint(cvRound(roi.width/2.0+roi.x),
                 cvRound(roi.height/2.0+roi.y));
}

//-------------------------------------------------------
// INPUT/OUTPUT FUNCTIONS
//-------------------------------------------------------
CV_INLINE
void cvPrintMinMax(CvMat * arr)
{
  double minval, maxval;
  CvPoint minloc, maxloc;
  cvMinMaxLoc(arr, &minval, &maxval, &minloc, &maxloc);
  fprintf(stderr,
          "\nminval: %f\nmaxval: %f\n"
          "minloc: (%d,%d)\nmaxloc: (%d,%d)\n",
          minval, maxval, minloc.x, minloc.y, maxloc.x, maxloc.y);
}

/** 
 * 
 * 
 * @param fp 
 * @param arr 
 */
CV_INLINE
void cvPrintEx(FILE * fp, const CvArr * arr)
{
  CvMat * raw = NULL;
  if (CV_IS_MAT(arr)) {raw=(CvMat*)arr;}
  else {CvMat hdr;raw=cvGetMat(arr, &hdr);}
  
  // FILE * fp = fopen(fn, "w");
  if (!fp) {fprintf(stderr, "Error: open file error!\n");return;}
  if (cvGetElemType(arr)==CV_32F || cvGetElemType(arr)==CV_64F){
    for (int i = 0; i < raw->rows; i++){
      for (int j = 0; j < raw->cols; j++){
        fprintf(fp, "%f ", cvmGet(raw, i, j));
      }fprintf(fp, "\n");
    }
  }else if (cvGetElemType(arr)==CV_8U){
    for (int i = 0; i < raw->rows; i++){
      for (int j = 0; j < raw->cols; j++){
        fprintf(fp, "%3d ", CV_MAT_ELEM(*raw, uchar, i, j));
      }fprintf(fp, "\n");
    }
  }else if (cvGetElemType(arr)==CV_32SC2){ // print contour
    for (int i = 0; i < raw->rows; i++){
      for (int j = 0; j < raw->cols; j++){
        fprintf(fp, "(%d,%d) ",
                CV_MAT_ELEM(*raw, CvPoint, i, j).x,
                CV_MAT_ELEM(*raw, CvPoint, i, j).y);
      }fprintf(fp, "\n");
    }
  }else{
    fprintf(stderr, "Warning: Unsupported data type: %d\n"
            "\tCV_32F: %d\n"
            "\tCV_8U: %d\n"
            "\tCV_32SC2: %d\n", cvGetElemType(arr),
            CV_32F, CV_8U, CV_32SC2);
  }
  // fclose(fp);
}

CVAPI(void) cvPrintf(FILE * fp, const char * format, CvArr * arr,
                     CvRect roi=cvRect(0,0,0,0));

CV_INLINE
void cvScanf(FILE * fp, const char * format,
             int nrows, int ncols, CvArr * dst)
{
  CvMat header;
  CvMat * imgDst = cvGetMat(dst, &header);
  int type = CV_MAT_TYPE(imgDst->type);
  int tmpval;
  if (type==CV_8U){
    uchar bval;
    for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
        tmpval=fscanf(fp, format, &bval);
        CV_MAT_ELEM(*imgDst, uchar, i, j)=bval;
      }
    }
  }else if (type==CV_32S){
    int ival;
    for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
        tmpval=fscanf(fp, format, &ival);
        CV_MAT_ELEM(*imgDst, int, i, j) = ival;
      }
    }
  }else if (type==CV_32F){
    float fval;
    for (int i = 0; i < nrows; i++){
      for (int j = 0; j < ncols; j++){
        tmpval=fscanf(fp, format, &fval);
        CV_MAT_ELEM(*imgDst, float, i, j) = fval;
      }
    }
  }
}

CV_INLINE
void cvPrintROI(FILE * fp, const CvArr * arr, const CvRect roi)
{
  CvMat header;
  CvMat * mat = cvGetSubRect(arr, &header, roi);
  cvPrintEx(fp, mat);
}

CVAPI(void) cvPrintHeader(const char * fn, const CvArr * arr);


/** 
 * load .csv file as matrix data
 * 
 * @param fn    in:  static file name with .csv subfix
 * @param type  in:  expected data type stored in .csv file
 * 
 * @return out: matrix data that stored in the file
 */
CVAPI(CvMat *) icvLoadCSV(char * fn, int type);

/** 
 * load matlab generated csv files
 * 
 * @param fn        in:  CSV filename to be loaded
 * 
 * @return          data matrix 
 */
#if 0
CVAPI(CvMat*) cvLoadCSV(const char * fn);
#endif

/** 
 * 
 * 
 * @param fn 
 * @param arr 
 */
CV_INLINE
void cvSaveEx(const char * fn, const CvArr * arr)
{
  int len = strlen(fn);
  if (strncmp(fn+len-4, ".xml", 4)==0 || strncmp(fn+len-4, ".yml", 4)==0){
    cvSave(fn, arr);
  }else if ( (strncmp(fn+len-4, ".txt", 4)==0) ){
    FILE * fp = fopen(fn, "w");
    if (!fp) {fprintf(stderr, "Error: open file error!\n");return;}
    cvPrintEx(fp, arr);
    fclose(fp);
  }else{
    fprintf(stderr,
            "Warning: Unexpected file extenstion!\n");
  }
}

//------------------------------------------------------------------
// CONVERSION
//------------------------------------------------------------------

/** 
 * convert floating-type matrix to 8-bit matrix
 * 
 * @param src IN: floating type matrix to be converted
 * @param dst OUT: 8-bit matrix 
 */
CV_INLINE
void cvConvertScaleAbsEx(const CvArr * src, CvArr * dst)
{
  double minval, maxval; CvPoint minloc, maxloc;
  double maxdiff;
  IplImage * raw;
  CV_FUNCNAME("cvConvertScaleAbsEx");
  __BEGIN__;
  int type = cvGetElemType(src);
  CV_ASSERT(type==CV_16S || type==CV_32F || type==CV_64F);
  CV_ASSERT(cvGetElemType(dst)==CV_8U);

  cvMinMaxLoc(src, &minval, &maxval, &minloc, &maxloc);
  maxdiff = maxval - minval;
  IplImage rawHeader;
  raw = cvGetImage(src, &rawHeader);
  cvScale(raw, raw, 255.f/maxdiff, -minval);
  cvConvert(raw, dst);
  __END__;
}

/** 
 * convenient function for showing floating type image,
 * compatible with original `cvShowImage' function.
 * 
 * @param id    IN: named window
 * @param arr   OUT: matrix to be displayed in the window
 */
CvScalar cvHue2BGR( float hue );

CVAPI(void) cvShowImageEx(const char * id, const CvArr * arr,
                          const int cm = CV_CM_HSV);
CVAPI(void) cvSaveImageEx(const char * id, const CvArr * arr, CvRect roi);

#ifndef ANDROID
/** 
 * Enable exit on ESC keypress, stop on SPACE keypress,
 * frame by frame on any other keys
 * 
 * @param delay     IN: time delay to wait for the key press, default is 10.
 * 
 * @return          OUT: key press ASCII code
 */
CV_INLINE
int cvWaitKeyEx(const int delay CV_DEFAULT(10))
{
  static int dd = delay;
  // stop when delay is zero
  int key = (delay==0)?(cvWaitKey()):(cvWaitKey(dd)&0xff); 
  if (key==' '){dd=((dd==0)?delay:0);}
  return key;
}

#define CV_WAIT()                                                       \
  do { if ( (cvWaitKey(0)&0xff)==27 ) { exit(1); } } while ( false )
#define CV_WAIT2(ddelay)                                                \
  do { if ( (cvWaitKey((ddelay))&0xff)==27 ) { exit(1); } } while ( false )
#define CV_SHOW(img)                            \
  cvShowImageEx("Test", (img), CV_CM_GRAY); CV_WAIT()
#else
#define CV_SHOW(img) do {} while(false)
#endif // ANDROID

#if defined(__cplusplus) && !defined(ANDROID)
class CvWindowManager{
public:
  CvWindowManager(){cvNamedWindow("Test");}
  ~CvWindowManager(){cvDestroyWindow("Test");}
};
#endif

//-------------------------------------------------------
// SIMPLE CALCULATION FUNCTIONS
//-------------------------------------------------------

/**
 * Calculate distance between given points
 */
CV_INLINE
float cvDistance(CvPoint p1, CvPoint p2)
{
  float dx = p1.x-p2.x;
  float dy = p1.y-p2.y;
  return sqrt(dx*dx+dy*dy);
}

CV_INLINE
float cvDistance(CvPoint2D32f p1, CvPoint2D32f p2)
{
  float dx = p1.x-p2.x;
  float dy = p1.y-p2.y;
  return sqrt(dx*dx+dy*dy);
}

CV_INLINE
double cvDistance(CvPoint2D64f p1, CvPoint2D64f p2)
{
  double dx = p1.x-p2.x;
  double dy = p1.y-p2.y;
  return sqrt(dx*dx+dy*dy);
}

/** 
 * Normalize length of given vector,
 * make its length equals to 1 without changing its direction
 * 
 * @param p 
 */
CV_INLINE
void cvNormalizeS(CvPoint3D32f & p)
{
  float len = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
  p = cvPoint3D32f(p.x/len, p.y/len, p.z/len);
}

/** 
 * Calculate normal vector at given point
 * 
 * @param a     IN:  1st vector 
 * @param b     IN:  2nd vector at the same point
 * @param normalize IN:
 *        default value is set to false,
 *        indicating whether the output vector should be normalized
 * 
 * @return      OUT: normal vector 
 */
CV_INLINE
CvPoint3D32f cvCalcNormalS(const CvPoint3D32f a,
                           const CvPoint3D32f b,
                           const bool normalize CV_DEFAULT(false))
{
  CvPoint3D32f pt = cvPoint3D32f(a.y*b.z-a.z*b.y,
                                 a.z*b.x-a.x*b.z,
                                 a.x*b.y-a.y*b.x);
  if (normalize){ cvNormalizeS(pt); }
  return pt;
}

/** 
 * convenient function for cvDilate with inplace modification
 * 
 * @param im IN/OUT: image to be modified
 * @param ksize IN: the size of kernel
 */
CV_INLINE
void cvDilateEx(CvArr * im, const int ksize)
{
  IplConvKernel * kernel = NULL;
  {
    kernel =
        cvCreateStructuringElementEx(ksize*2+1, ksize*2+1,
                                     ksize, ksize,
                                     CV_SHAPE_ELLIPSE);
    cvDilate(im, im, kernel);
  }
  cvReleaseStructuringElement(&kernel);
}

CV_INLINE
void cvDilateEx(const CvArr * src, CvArr * dst, const int ksize)
{
  assert(ksize>0);
  IplConvKernel * kernel =
      cvCreateStructuringElementEx(ksize*2+1, ksize*2+1, ksize, ksize, CV_SHAPE_ELLIPSE);
  cvDilate(src, dst, kernel);
  cvReleaseStructuringElement(&kernel);
}

/** 
 * convenient function for cvErode with inplace modification
 * 
 * @param im IN:OUT: image to modified
 * @param ksize IN: the size of kernel
 */
CV_INLINE
void cvErodeEx(CvArr * im, const int ksize )
{
  IplConvKernel * kernel = NULL;
  {
    kernel =
        cvCreateStructuringElementEx(ksize*2+1, ksize*2+1,
                                     ksize, ksize,
                                     CV_SHAPE_ELLIPSE);
    cvErode(im, im, kernel);
  }
  cvReleaseStructuringElement(&kernel);
}

/** 
 * convenient function for Gaussian smoothing with given kernel size 3.
 * 
 * @param im    IN:OUT: image to be smoothed
 * @param ksize IN:     kernel size of gaussian
 */
CV_INLINE
void cvSmoothEx(IplImage *& im, const int ksize CV_DEFAULT(3))
{
  int kk = (ksize%2==0)?(ksize+1):ksize;
  assert(kk>2); // kernel size must be odd
  cvSmooth(im, im, CV_GAUSSIAN, kk, kk);
}

/** 
 * In-place ...
 * 
 * @param im 
 */
CV_INLINE
void cvEqualizeHistEx(CvArr * im)
{
  IplImage header;
  IplImage * rawImage = cvGetImage(im, &header);
  if (cvGetElemType(rawImage)==CV_32FC1){
    IplImage * rawImage8u =
        cvCreateImage(cvGetSize(rawImage), IPL_DEPTH_8U, 1);
    cvConvertScaleAbsEx(rawImage, rawImage8u);
    cvEqualizeHist(rawImage8u, rawImage8u);
    cvConvert(rawImage8u, rawImage);
    cvReleaseImageEx(rawImage8u);
  }else if (cvGetElemType(rawImage)==CV_8UC1){
    cvEqualizeHist(rawImage, rawImage);
  }else{
    fprintf(stderr, "Warning: data type is not supported !!\n");
    assert(false);
  }
}

/** 
 * 
 * 
 * @param x 
 * @param y 
 * @param mag 
 */
CV_INLINE
void cvCartToPolarEx(const CvArr * x, const CvArr * y, CvArr * mag)
{
  if (cvGetElemType(x)==CV_8U && cvGetElemType(mag)==CV_32F){
    IplImage * xx =
        cvCreateImage(cvGetSize(x), IPL_DEPTH_32F, 1);
    IplImage * yy =
        cvCreateImage(cvGetSize(y), IPL_DEPTH_32F, 1);
    cvConvert(x, xx);
    cvConvert(y, yy);
    cvCartToPolar(xx, yy, mag);
    cvReleaseImageEx(xx);
    cvReleaseImageEx(yy);
  }else{
    cvCartToPolar(x, y, mag);
  }
}

//-------------------------------------------------------
// ENHENCED DRAWING FUNCTIONS
//-------------------------------------------------------

/** 
 * Paint text on image without initialing font type
 * 
 * @param disp IN: the image to be painted onto
 * @param str IN: the text to be drawn
 * @param pt  IN: the position to put the text
 * @param color IN: the text color information
 *                  (set to be BLUE by default)
 */
CV_INLINE
void cvPutTextEx(CvArr * _disp, const char* str,
                 const CvPoint pt, 
                 const CvScalar color = CV_RGB(0,0,255),
                 const float scale = 0.5,
                 const int thickness = 1)
{
  CV_FUNCNAME("cvPutTextEx");
  __BEGIN__;
  CvFont font;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX, scale, scale, 0, thickness);
  IplImage * disp, disp_stub;
  if (!CV_IS_MAT(_disp))
    disp = (IplImage*)_disp;
  else
    disp = cvGetImage(_disp, &disp_stub);
  cvPutText(disp, str, pt, &font, color);
  __END__;
}

/** 
 * 
 * 
 * @param box 
 * @param pt 
 */
CV_INLINE
void cvBoxPoints32s( CvBox2D box, CvPoint pt[4] )
{
  // double angle = box.angle*CV_PI/180.;
  double angle = box.angle*0.0174532;
  double a = (double)cos(angle)*.5;
  double b = (double)sin(angle)*.5;
  double aw = a*box.size.width;
  double bh = b*box.size.height;
  double bw = b*box.size.width;
  double ah = a*box.size.height;
  float x = box.center.x;
  float y = box.center.y;

  pt[0].x = (x - aw - bh);
  pt[0].y = (y + bw - ah);
  pt[1].x = (x + aw - bh);
  pt[1].y = (y - bw - ah);
  pt[2].x = (2.*x - pt[0].x);
  pt[2].y = (2.*y - pt[0].y);
  pt[3].x = (2.*x - pt[1].x);
  pt[3].y = (2.*y - pt[1].y);
}

/** 
 * 
 * 
 * @param box 
 * @param pt 
 */
CV_INLINE
void cvBoxPoints32f( CvBox2D box, CvPoint2D32f pt[4] )
{
  double angle = box.angle*CV_PI/180.;
  float a = (float)cos(angle)*0.5f;
  float b = (float)sin(angle)*0.5f;

  pt[0].x = box.center.x - a*box.size.width - b*box.size.height;
  pt[0].y = box.center.y + b*box.size.width - a*box.size.height;
  pt[1].x = box.center.x + a*box.size.width - b*box.size.height;
  pt[1].y = box.center.y - b*box.size.width - a*box.size.height;
  pt[2].x = 2*box.center.x - pt[0].x;
  pt[2].y = 2*box.center.y - pt[0].y;
  pt[3].x = 2*box.center.x - pt[1].x;
  pt[3].y = 2*box.center.y - pt[1].y;
}

CV_INLINE
CvBox2D cvBox2D(const float x, const float y,
                const float w, const float h, const float a)
{
  CvBox2D box;

  box.center = cvPoint2D32f(x, y);
  box.size.width = w;
  box.size.height = h;
  box.angle = a;

  return box;
}

CV_INLINE
CvBox2D cvBox2DFromRect(const CvRect rect)
{
  CvBox2D box;

  box.center.x = rect.x+rect.width/2.f;
  box.center.y = rect.y+rect.height/2.f;
  box.size.width  = rect.width;
  box.size.height = rect.height;
  box.angle = 0;

  return box;
}

CV_INLINE
CvRect cvBox2DToRect(const CvBox2D box)
{
#if 1
  CvPoint pts[4]; cvBoxPoints32s(box, pts);
  CvPoint lt =
      cvPoint(MIN(pts[0].x,pts[3].x),
              MIN(pts[0].y,pts[1].y));
  CvPoint rb =
      cvPoint(MAX(pts[1].x,pts[2].x),
              MAX(pts[2].y,pts[3].y));
  return cvRect(lt.x, lt.y, rb.x-lt.x, rb.y-lt.y);

#else
  CvPoint2D32f pts[4];
  double angle = (360-box.angle)*CV_PI/180.;
  float a = (float)cos(angle)*0.5f;
  float b = (float)sin(angle)*0.5f;

  pts[0].x = box.center.x - a*box.size.width - b*box.size.height;
  pts[0].y = box.center.y + b*box.size.width - a*box.size.height;
  pts[1].x = box.center.x + a*box.size.width - b*box.size.height;
  pts[1].y = box.center.y - b*box.size.width - a*box.size.height;
  pts[2].x = 2*box.center.x - pts[0].x;
  pts[2].y = 2*box.center.y - pts[0].y;
  pts[3].x = 2*box.center.x - pts[1].x;
  pts[3].y = 2*box.center.y - pts[1].y;

  CvPoint lt =
      cvPoint(cvCeil((pts[0].x+pts[3].x)/2.f),
              cvCeil((pts[0].y+pts[1].y)/2.f));
  CvPoint rb =
      cvPoint(cvCeil((pts[1].x+pts[2].x)/2.f),
              cvCeil((pts[2].y+pts[3].y)/2.f));
  return cvRect(lt.x, lt.y, rb.x-lt.x, rb.y-lt.y);
#endif
}

/** 
 * 
 * 
 * @param img 
 * @param box 
 * @param color 
 * @param thinkness 
 */
CV_INLINE
void cvBoxRectangle(CvArr * img, CvBox2D box,
                    CvScalar color, int thinkness=1)
{
  CvPoint pts[4];
  cvBoxPoints32s(box, pts);

  cvLine(img, pts[0], pts[1], color, thinkness);
  cvLine(img, pts[2], pts[1], color, thinkness);
  cvLine(img, pts[2], pts[3], color, thinkness);
  cvLine(img, pts[0], pts[3], color, thinkness);
}

/** 
 * Draw text label from top-left corner
 * 
 * @param disp 
 * @param str 
 * @param line 
 */
CV_INLINE
void cvDrawLabel(CvArr * _disp, const char * str,
                 const CvScalar color = CV_RGB(0,0,255),
                 const int line = 0,
                 const float thinkness = 1,
                 const float scale = 1
                 )
{
  // IplImage * disp, disp_stub;
  // if (!CV_IS_MAT(_disp))
  //   disp = (IplImage*)_disp;
  // else
  //   disp = cvGetImage(_disp, &disp_stub);
  // cvPutTextEx(disp, str, cvPoint(10, 10*line+15), color,0.5,thinkness);
  //CvPoint loc=cvPoint(10,10);
  cvPutTextEx(_disp, str, cvPoint(10*scale, 10*scale*line+15*scale), color,scale*0.5,thinkness);
}

/** 
 * Draw frame index at bottum-left corner of given image frame
 * 
 * @param img           IN/OUT: image to be painted
 * @param framecounter  IN:     frame index
 * @param color         IN:     color of the text
 */
CV_INLINE
void cvDrawFrameCount(CvArr * img, const int framecounter,
                      CvScalar color CV_DEFAULT(cvScalarAll(255)))
{
  CV_FUNCNAME("cvDrawFrameCount");

  __BEGIN__;
  char stateStr[CV_MAXSTRLEN];
  sprintf(stateStr, "frame: %d", framecounter);
  cvPutTextEx(img, stateStr, cvPoint(5, cvGetSize(img).height-20),
              color, 0.5, 1);
  __END__;
}

/** 
 * hue to RGB conversion :
 * coverts a given hue value to a RGB triplet for display
 *
 * Notes: taken from OpenCV 1.0 camshiftdemo.c example
 * 
 * @param hue - hue value in range 0 to 180 (OpenCV implementation of HSV)
 * @return value - CvScalar as RGB triple
 */
inline CvScalar cvHue2BGR( float hue )
{
  int rgb[3], p, sector;
  static const int sector_data[][3]=
      {{0,2,1}, {1,2,0}, {1,0,2}, {2,0,1}, {2,1,0}, {0,1,2}};
  // for 0~180 representing 360
  // hue *= 0.033333333333333333333333333333333f; 
  hue *= 0.016666666666666666666666666666666f;
  sector = cvFloor(hue);
  p = cvRound(255*(hue - sector));
  p ^= sector & 1 ? 255 : 0;

  rgb[sector_data[sector][0]] = 255;
  rgb[sector_data[sector][1]] = 0;
  rgb[sector_data[sector][2]] = p;

  return cvScalar(rgb[2], rgb[1], rgb[0],0);
}


/** 
 * map CV_8U grayscale image to color image
 * with assigned colormap lookup table
 * 
 * @param _src in: grayscale image, CV_8U
 * @param _dst out: color image, CV_8UC3
 * @param _lut in: 256x3 CV_8U colormap look-up table
 */
CVAPI(void) cvColorMapLUT(CvArr * _src, CvArr * _dst, CvArr * _lut);

/** 
 * 
 * 
 * @param img 
 * @param contour 
 * @param color 
 * @param filled
 */
CV_INLINE
void cvDrawContourEx(CvArr * _img, const CvArr * _contour,
                     // white contour boundary by default
                     const CvScalar color CV_DEFAULT(cvScalarAll(255)),
                     // not filled by default
                     const int fillflag CV_DEFAULT(false)) 
{
  IplImage imgHeader; CvMat contourHeader;
  IplImage * im = cvGetImage(_img, &imgHeader);
  CvMat * contour = NULL;
  if (cvGetElemType(_contour)==CV_32SC2){
    contour = cvGetMat(_contour, &contourHeader);
  }else{
    // convert type of contour in case
    // contour is not described by integer
    CvMat * contour_orig = cvGetMat(_contour, &contourHeader);
    contour =
        cvCreateMat(contour_orig->rows, contour_orig->cols, CV_32SC2);
    cvConvert(contour_orig, contour);
  }
  if (contour->cols==0 || contour->rows!=1){return;}
  if (fillflag)
  {
    CvMat * icontour =
        cvCreateMat(contour->rows, contour->cols, CV_32SC2);
    cvConvert(contour, icontour);
    CvMemStorage * storage = cvCreateMemStorage(0);
    CvSeq * seq = cvCreateSeq(CV_SEQ_KIND_CURVE|CV_32SC2,
                              sizeof(CvContour),
                              sizeof(CvPoint),
                              storage);
    cvSeqPushMulti(seq, (CvPoint*)icontour->data.ptr, contour->cols);
    cvDrawContours(im, seq, color, CV_RGB(255,0,0), 1, CV_FILLED);
    cvReleaseMemStorage(&storage);
    cvReleaseMatEx(icontour);
  }
  else
  {
    if (cvGetElemType(contour)==CV_32SC2){
      for (int i = 0; i < (int)contour->cols-1; i++) {
        cvLine(im,
               CV_MAT_ELEM(*contour, CvPoint, 0, i),
               CV_MAT_ELEM(*contour, CvPoint, 0, i+1),
               color);
      }
      cvLine(im,
             CV_MAT_ELEM(*contour, CvPoint, 0, 0),
             CV_MAT_ELEM(*contour, CvPoint, 0, contour->cols-1),
             color);
    }else if (cvGetElemType(contour)==CV_64FC2){
      for (int i = 0; i < (int)contour->cols-1; i++) {
        cvLine(im,
               cvPoint(CV_MAT_ELEM(*contour, CvPoint2D64f, 0, i).x,
                       CV_MAT_ELEM(*contour, CvPoint2D64f, 0, i).y),
               cvPoint(CV_MAT_ELEM(*contour, CvPoint2D64f, 0, i+1).x,
                       CV_MAT_ELEM(*contour, CvPoint2D64f, 0, i+1).y),
               color);
      }
      cvLine(im,
             cvPoint(CV_MAT_ELEM(*contour, CvPoint2D64f, 0, 0).x,
                     CV_MAT_ELEM(*contour, CvPoint2D64f, 0, 0).y),
             cvPoint(CV_MAT_ELEM(*contour, CvPoint2D64f, 0,
                                 contour->cols-1).x,
                     CV_MAT_ELEM(*contour, CvPoint2D64f, 0,
                                 contour->cols-1).y),
             color);
    }else{
      fprintf(stderr, "Warning: data type not supported !\n");
    }
  }
  if (cvGetElemType(_contour)!=CV_32SC2)
    cvReleaseMatEx(contour);
}

/** 
 * 
 * 
 * @param img 
 * @param landmarks 
 * 
 * @return 
 */
// CVAPI(void)
CV_INLINE
void cvDrawLandmarks(CvArr * _img, const CvArr * _landmarks,
                     CvScalar color CV_DEFAULT(CV_RGB(0,0,255)))
{
  CV_FUNCNAME("cvDrawLandmarks");
  IplImage imgheader; CvMat landmarkhdr;
  IplImage * img = cvGetImage(_img, &imgheader);
  CvMat * landmarks = cvGetMat(_landmarks, &landmarkhdr);
  __BEGIN__;

  if (landmarks->rows==1){
	  CV_ASSERT(CV_MAT_TYPE(landmarks->type)==CV_64F);
	  const int npoints = landmarks->cols/2;
	  for (int i = 0; i < npoints*2-2; i+=2){
		  CvPoint pt1 =
			  cvPoint( CV_MAT_ELEM(*landmarks, double, 0, i),
			  CV_MAT_ELEM(*landmarks, double, 0, i+1));
		  CvPoint pt2 =
			  cvPoint( CV_MAT_ELEM(*landmarks, double, 0, i+2),
			  CV_MAT_ELEM(*landmarks, double, 0, i+3) );
		  cvCircle(img, pt1,2,color,-1);
		  cvLine(img, pt1, pt2, color);
	  }
  }
  else if (landmarks->cols==1)
  {
	  CV_ASSERT(CV_MAT_TYPE(landmarks->type)==CV_32S);
	  for (int i = 0; i < landmarks->rows-1; i++){
		  CvPoint pt1 =
			  cvPoint( CV_MAT_ELEM(*landmarks, int, i, 0),
			  CV_MAT_ELEM(*landmarks, int, i, 1));
		  CvPoint pt2 =
			  cvPoint( CV_MAT_ELEM(*landmarks, int, i+1, 0),
			  CV_MAT_ELEM(*landmarks, int, i+1, 1) );
		  cvLine(img, pt1, pt2, color);
	  }
  }else if (landmarks->rows==2){
	  //if (landmarks->cols==2) {cvTranspose(landmarks, landmarks);}
	  if ( CV_MAT_TYPE(landmarks->type)==CV_32F ||
	       CV_MAT_TYPE(landmarks->type)==CV_64F )
	  {
		  for (int i = 0; i < landmarks->cols; i++){
			  CvPoint pt =
				  cvPoint(cvmGet(landmarks, 0, i), cvmGet(landmarks, 1, i));
			  cvCircle(img,pt,2,color,-1);
		  }
		  for (int i = 0; i < landmarks->cols-1; i++){
			  CvPoint pt1 =
				  cvPoint(cvmGet(landmarks, 0, i), cvmGet(landmarks, 1, i));
			  CvPoint pt2 =
				  cvPoint(cvmGet(landmarks, 0, i+1), cvmGet(landmarks, 1, i+1));
			  cvLine(img, pt1, pt2, color);
		  }
	  }else if ( CV_MAT_TYPE(landmarks->type)==CV_32S ){
		  for (int i = 0; i < landmarks->cols; i++){
			  CvPoint pt =
				  cvPoint(CV_MAT_ELEM(*landmarks, int, 0, i), CV_MAT_ELEM(*landmarks, int, 1, i));
			  cvCircle(img,pt,2,color,-1);
		  }
	  }
  }else if (landmarks->cols==2 && CV_MAT_TYPE(landmarks->type)==CV_32S ){
	  for (int i = 0; i < landmarks->rows; i++){
		  CvPoint pt =
			  cvPoint(CV_MAT_ELEM(*landmarks, int, i, 0), CV_MAT_ELEM(*landmarks, int, i, 1));
		  cvCircle(img,pt,2,color,-1);
	  }
  }else{
#if CV_MAJOR_VERSION==1
	  CV_ERROR(CV_BadImageSize, 
               "number of rows in landmark should be either 1 or 2; "
               "and data content should be 32-bit or 64-bit floating point.");
#endif
  }
  __END__;
}

/** 
 * vertically combine images
 * 
 * @param src1 IN: image to be put on the top
 * @param src2 IN: image to be put on the bottom
 * 
 * @return OUT: combined image with same depth and nChannels
 */
CV_INLINE
IplImage * cvCombineImagesV(const IplImage * src1,
                            const IplImage * src2, IplImage * dst)
{
  assert(src1->width==src2->width);
  assert(src1->height==src2->height);
  assert(src1->origin==src2->origin);
  assert(src1->depth==src2->depth && src2->depth==8);
  assert(src1->nChannels==src2->nChannels);
  assert(src2->nChannels==3 || src2->nChannels==1);

  CvSize size = cvSize(src1->width, src1->height*2);
  dst = cvCreateImage(size, IPL_DEPTH_8U, src2->nChannels);

  memcpy(dst->imageData, src1->imageData,
         sizeof(char)*src1->widthStep*src1->height);
  memcpy(dst->imageData+
         sizeof(char)*src1->widthStep*src1->height,
         src2->imageData,
         sizeof(char)*src2->widthStep*src2->height);
	
  //return dst;
}

/** 
 * combine images horizontally
 * 
 * @param src1 IN: image to be put on the top
 * @param src2 IN: image to be put on the bottom
 * 
 * @return OUT: combined image with same depth and nChannels
 */
CV_INLINE
IplImage * cvCombineImagesH(const IplImage * const src1,
                                   const IplImage * const src2)
{
  assert(src1->width==src2->width);
  assert(src1->height==src2->height);
  assert(src1->origin==src2->origin);
  assert(src1->depth==src2->depth && src2->depth==8);
  assert(src1->nChannels==src2->nChannels);
  assert(src2->nChannels==3 || src2->nChannels==1);

  CvSize size = cvSize(src1->width*2, src1->height);
  IplImage * dst = cvCreateImage(size, IPL_DEPTH_8U, src2->nChannels);

  for (int i = 0; i < size.height; i++){
    memcpy(dst->imageData  + i * src1->widthStep * 2,
           src1->imageData + i * src1->widthStep,
           sizeof(uchar) * src1->widthStep);
    memcpy(dst->imageData  + i * src1->widthStep * 2 + src1->widthStep,
           src2->imageData + i * src1->widthStep,
           sizeof(uchar) * src1->widthStep);
  }
  return dst;
}

//-------------------------------------------------------
// COMPLICATED CALCULATING FUNCTIONS
//-------------------------------------------------------

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
CVAPI(void) cvCalcGradient(const CvArr * src,
                           CvArr * dst,
                           const int ksize CV_DEFAULT(1));

template <typename DataType>
void cvCalcGradient(const CvArr * _src,
                    CvArr * _dx = NULL,
                    CvArr * _dy = NULL,
                    CvArr * _mag = NULL); // magnitude (optional)

/*
 * This is a sweep-and-update Euclidean distance transform of
 * a binary image. All positive pixels are considered object
 * pixels, zero or negative pixels are treated as background.
 *
 * By Stefan Gustavson (stefan.gustavson@gmail.com).
 *
 * Originally written in 1994, based on paper-only descriptions
 * of the SSED8 algorithm, invented by Per-Erik Danielsson
 * and improved by Ingemar Ragnemalm. This is a classic algorithm
 * with roots in the 1980s, still very good for the 2D case.
 *
 * Updated in 2004 to treat pixels at image edges correctly,
 * and to improve code readability.
 *
 */
/** 
 * Euclidean distance transform of a binary image,
 * Computation performed upon 'short' values
 * 
 * @param src       IN: positive pixels are treated as background
 * @param distxArr  OUT: 
 * @param distyArr  OUT: 
 */
CVAPI(void) cvCalcEuclideanDistance(
    const CvArr * src,      // positive pixels are treated as background
    CvArr * distxArr, CvArr * distyArr);

/** 
 * perform point-polygon test for every point in given image frame
 * (Warning: this is very SLOW!!!)
 * 
 * @param contour   IN:  1xN CV_32FC2, indicates contour
 * @param dst       OUT: signed distance function
 * 
 * @return OUT: status information
 */
// CVAPI(CVStatus) cvCalcSignedDistance(const CvArr * contour, CvArr * dst);
CVAPI(CVStatus) cvCalcDistTransform(const CvArr * src, CvArr * dst);

/** 
 * Extract (normalized) fourier descriptors,
 * Size of output fourier descriptors is determined
 * by the number of columns in destinate matrix
 * 
 * @param contour   IN:  1xN CV_32FC2 , indicating contour of shape
 * @param fdesc     OUT: 1xK CV_32FC2 , indicating fourier descriptors
 */
CV_INLINE
void cvExtractFourierDescriptors(
    const CvArr * contour,       // 1xN CV_32FC2 
    CvArr * fdesc                // 1xK CV_32FC2 
                                 )
{
  assert(contour); assert(fdesc);
  CvMat matContourHeader, matFDescHeader;
  CvMat * matContour = cvGetMat(contour, &matContourHeader);
  CvMat * matFDesc = cvGetMat(fdesc, &matFDescHeader); cvZero(matFDesc);
  assert(matContour->rows==1);
  assert(matFDesc->rows==1);

  const int conlen = matContour->cols;  // length of contour
  const int fdsize = matFDesc->cols;    // size of fourier descriptors
  assert(fdsize%2==0);                  // assume fdsize is even

  CvMat * matFourier =
      cvCreateMat(matContour->rows, matContour->cols, CV_32FC2);

  // perform fourier transform 
  cvDFT(matContour, matFourier, CV_DXT_FORWARD);

  // centering - set first element to zero
  CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, 0) = cvPoint2D32f(0,0);

  // scaling - divide by magnitude of 2nd element
  {
#if 0
    CvPoint2D32f p =
        // extract last item of the vector, as scaling factor
        CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, matFourier->cols-1);
#else
    CvPoint2D32f p =
        // extract 2nd item of the vector, as scaling factor
        CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, 1);
#endif
    float scale_coeff = sqrt((p.x*p.x)+(p.y*p.y));
    cvScale(matFourier, matFourier, 1.f/scale_coeff);
  }

  // cropping - get correct size of fd (high frequency section)
  int halflen = (conlen+1)/2-1;
  if (conlen<fdsize) // case of contour length less than fd size
  {
    for (int i = 0; i < halflen; i++){
      // front to middle
      CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, i) =
          CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, i+1);
      // back to middle
      CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, fdsize-1-i) =
          CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, conlen-1-i);
    }
  }else{
    for (int i = 0; i < fdsize/2; i++){
      // front to middle
      CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, i) =
          CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, i+1);
      // back to middle
      CV_MAT_ELEM(*matFDesc, CvPoint2D32f, 0, fdsize-1-i) =
          CV_MAT_ELEM(*matFourier, CvPoint2D32f, 0, conlen-1-i);
    }
  }
  cvReleaseMatEx(matFourier);
}

/*
 * cvWiener2 -- A Wiener 2D Filter implementation for OpenCV
 *  Author: Ray Juang  / rayver {_at_} hkn {/_dot_/} berkeley (_dot_) edu
 *  Date: 12.1.2006
 *
 * Modified 1.5.2007 (bug fix --
 *   Forgot to subtract off local mean from local variance estimate.
 *   (Credits to Kamal Ranaweera for the find)
 *
 * Modified 1.21.2007 (bug fix --
 *   OpenCV's documentation claims that the default anchor for
 * cvFilter2D is center of kernel.
 *   This seems to be a lie -- the center has to be explicitly stated
 * Usage:
 *  IplImage *tmp = cvLoadImage(argv[1]);
 *  IplImage *tmp2 =
 *      cvCreateImage(cvSize(tmp->width, tmp->height), IPL_DEPTH_8U, 1);
 *	cvCvtColor(tmp, tmp2, CV_RGB2GRAY);
 *  cvWiener2(tmp2, tmp2, 5, 5);
*/
CV_INLINE
void cvWiener2( const void* srcArr, void* dstArr,
                int szWindowX, int szWindowY )
{
  CV_FUNCNAME( "cvWiener2" );

  int nRows;
  int nCols;
  CvMat *p_kernel = NULL;
  CvMat srcStub, *srcMat = NULL;
  CvMat *p_tmpMat1, *p_tmpMat2, *p_tmpMat3, *p_tmpMat4;
  double noise_power;

  __BEGIN__;

  //// DO CHECKING ////

  if ( srcArr == NULL) {
#if CV_MAJOR_VERSION==1    
    CV_ERROR( CV_StsNullPtr, "Source array null" );
#endif
  }
  if ( dstArr == NULL) {
#if CV_MAJOR_VERSION==1    
    CV_ERROR( CV_StsNullPtr, "Dest. array null" );
#endif
  }

  nRows = szWindowY;
  nCols = szWindowX;

  p_kernel = cvCreateMat( nRows, nCols, CV_32F );
#if CV_MAJOR_VERSION==1
  CV_CALL( cvSet( p_kernel, cvScalar( 1.0 / (double) (nRows * nCols)) ) );
#else
  cvSet( p_kernel, cvScalar( 1.0 / (double) (nRows * nCols)) );
#endif
  //Convert to matrices
  srcMat = (CvMat*) srcArr;

  if ( !CV_IS_MAT(srcArr) ) {
#if CV_MAJOR_VERSION==1
    CV_CALL ( srcMat = cvGetMat(srcMat, &srcStub, 0, 1) ) ;
#else
    srcMat = cvGetMat(srcMat, &srcStub, 0, 1);
#endif
  }

  //Now create a temporary holding matrix
  p_tmpMat1 =
      cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
  p_tmpMat2 =
      cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
  p_tmpMat3 =
      cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));
  p_tmpMat4 =
      cvCreateMat(srcMat->rows, srcMat->cols, CV_MAT_TYPE(srcMat->type));

  //Local mean of input
  //localMean
  cvFilter2D( srcMat, p_tmpMat1, p_kernel, cvPoint(nCols/2, nRows/2)); 

  //Local variance of input
  cvMul( srcMat, srcMat, p_tmpMat2);	//in^2
  cvFilter2D( p_tmpMat2, p_tmpMat3, p_kernel, cvPoint(nCols/2, nRows/2));

  //Subtract off local_mean^2 from local variance
  //localMean^2
  cvMul( p_tmpMat1, p_tmpMat1, p_tmpMat4 ); 
  //filter(in^2) - localMean^2 ==> localVariance
  cvSub( p_tmpMat3, p_tmpMat4, p_tmpMat3 ); 

  //Estimate noise power
  //noise_power = cvMean(p_tmpMat3, 0);

  // result = local_mean  + ( max(0, localVar - noise) ./
  //                          max(localVar, noise)) .* (in - local_mean)

  cvSub ( srcMat, p_tmpMat1, dstArr);		     //in - local_mean
  cvMaxS( p_tmpMat3, noise_power, p_tmpMat2 ); //max(localVar, noise)

  //localVar - noise
  cvAddS( p_tmpMat3, cvScalar(-noise_power), p_tmpMat3 ); 
  cvMaxS( p_tmpMat3, 0, p_tmpMat3 ); // max(0, localVar - noise)

  //max(0, localVar-noise) / max(localVar, noise)
  cvDiv ( p_tmpMat3, p_tmpMat2, p_tmpMat3 );  
  cvMul ( p_tmpMat3, dstArr, dstArr );
  cvAdd ( dstArr, p_tmpMat1, dstArr );

  cvReleaseMat( &p_kernel  );
  cvReleaseMat( &p_tmpMat1 );
  cvReleaseMat( &p_tmpMat2 );
  cvReleaseMat( &p_tmpMat3 );
  cvReleaseMat( &p_tmpMat4 );

  __END__;
}

/**
 *****************************************************************
 **************** BAUM-WELCH ALGORITHM MATLAB CODE ***************
 *****************************************************************

 function [tr,E] = hmmestimate(seq,states)
 tr = [];  E = []; 
 seqLen = length(seq); 
 uniqueSymbols = unique(seq); 
 uniqueStates = unique(states); 
 numSymbols = length(uniqueSymbols); 
 numStates = length(uniqueStates); 
 tr = zeros(numStates); 
 E = zeros(numStates, numSymbols); 
 
 for count = 1:seqLen-1  % count up the transitions from the state path 
   tr(states(count),states(count+1)) =
     tr(states(count),states(count+1)) + 1; 
 end 
 for count = 1:seqLen    % count up the emissions for each state 
   E(states(count),seq(count)) = E(states(count),seq(count)) + 1; 
 end 
 trRowSum = sum(tr,2); ERowSum = sum(E,2); 
 trRowSum(trRowSum == 0) = -inf; 
 ERowSum(ERowSum == 0) = -inf; 
 tr = tr./repmat(trRowSum,1,numStates); 
 E = E./repmat(ERowSum,1,numSymbols); 
 
 * ****************************************************************
 */
CV_INLINE
void cvHMM2_EstimateParameters(
    /* input */ const CvArr * _seq, const CvArr * _states,
    /* output*/ CvArr * _trans, CvArr * _emit)
{
  CV_FUNCNAME("cvHMM2_EstimateParameters");
  __BEGIN__;
  
  CvMat matSeqHeader, matStatesHeader, matTransHeader, matEmitHeader;
  CvMat * matSeq = cvGetMat(_seq, &matSeqHeader);
  CvMat * matStates = cvGetMat(_states, &matStatesHeader);
  CvMat * matTrans = cvGetMat(_trans, &matTransHeader);
  CvMat * matEmit  = cvGetMat(_emit,  &matEmitHeader);

  const int seqLen = matSeq->rows;
  const int numSymbols = matEmit->cols;
  const int numStates  = matTrans->cols;

  assert(matSeq->rows==matStates->rows);
  assert(matSeq->cols==matStates->cols);
  assert(matTrans->rows==matTrans->cols);
  assert(matEmit->rows==numStates && matEmit->cols==numSymbols);

  cvZero(matTrans); cvZero(matEmit);
  
  for (int i = 0; i < seqLen-1; i++){
    CV_MAT_ELEM(*matTrans, float,
                matStates->data.ptr[i],
                matStates->data.ptr[i+1]) =
        CV_MAT_ELEM(*matTrans, float,
                    matStates->data.ptr[i],
                    matStates->data.ptr[i+1]) + 1.0f;
  }
  for (int i = 0; i < seqLen; i++){
    CV_MAT_ELEM(*matEmit, float,
                matStates->data.ptr[i],
                matSeq->data.ptr[i]) =
        CV_MAT_ELEM(*matEmit, float,
                    matStates->data.ptr[i],
                    matSeq->data.ptr[i]) + 1.0f;
  }

  CvMat * trRowSum = cvCreateMat(1, numStates, CV_32FC1);
  CvMat * ERowSum =  cvCreateMat(1, numStates, CV_32FC1);
  cvZero(trRowSum); cvZero(ERowSum);

  for (int i = 0; i < numStates; i++){
    assert(trRowSum->data.fl[i]>-0.001f &&
              trRowSum->data.fl[i]<0.001f);
    for (int j = 0; j < numStates; j++){
      trRowSum->data.fl[i] += CV_MAT_ELEM(*matTrans, float, i, j);
    }
    if (trRowSum->data.fl[i]>-0.001f && trRowSum->data.fl[i]<0.001f){
      trRowSum->data.fl[i] = HUGE_VAL;
    }

    for (int j = 0; j < numStates; j++){
      CV_MAT_ELEM(*matTrans, float, i, j) =
          CV_MAT_ELEM(*matTrans, float, i, j)/trRowSum->data.fl[i];
    }
  }
#ifdef _DEBUG
  // vector<float> vTransRowSum = cvMat2STLvectorf(trRowSum, 0);
  if (0)
  {
	  fprintf(stderr, "trans_probability:\n");
	  for (int i = 0 ; i < matTrans->rows; i++){
		  for (int j = 0; j < matTrans->cols; j++){
			  fprintf(stderr, "%.3f ", CV_MAT_ELEM(*matTrans, float, i, j));
		  }fprintf(stderr, "\n");
	  }fprintf(stderr, "\n");
  }
#endif
  for (int i = 0; i < numStates; i++){
    assert(ERowSum->data.fl[i]>-0.001f &&
           ERowSum->data.fl[i]<0.001f);
    for (int j = 0; j < numSymbols; j++){
      ERowSum->data.fl[i] += CV_MAT_ELEM(*matEmit, float, i, j);
    }
    if (ERowSum->data.fl[i]>-0.001f && ERowSum->data.fl[i]<0.001f){
      ERowSum->data.fl[i] = HUGE_VAL;
    }
    
    for (int j = 0; j < numSymbols; j++){
      CV_MAT_ELEM(*matEmit, float, i, j) =
          CV_MAT_ELEM(*matEmit, float, i, j)/ERowSum->data.fl[i];
    }
  }

  cvReleaseMatEx(trRowSum);
  cvReleaseMatEx(ERowSum);

  __END__; 
}

/** 
 * 
 * 
 * @param curr IN: 
 * @param next IN:
 * @param points IN:OUT:
 *               CV_32SC2 landmark points to track using optical flow
 * @param winsize IN: windows size for optical flow
 * @param nlevels IN: number of pyramid levels
 * 
 * @return OUT: error code
 */
CVStatus cvOpticalFlowPointTrack(
    const CvArr * curr,                    // current grayscale frame
    const CvArr * next,                    // next frame
    CvArr * points,                        // CV_32SC2 points to track
    const CvSize winsize CV_DEFAULT(cvSize(11,11)),
    const int nlevels CV_DEFAULT(3)        // number of pyramid levels
                                 );

/** 
 * Align shapes, for getting mean shape from a list of shapes for training
 *
 * Reference:
 *     Active shape models - their training and applications, Appendix-A
 * 
 * @param shape0        IN: 1st shape for comparison
 * @param shape1        IN: 2nd shape for comparison
 * @param scale         OUT: scale factor from 1st shape to 2nd
 * @param theta         OUT: rotation factor
 * @param tx            OUT: x-coord translation factor
 * @param ty            OUT: y-coord translation factor
 * @param use_radius    IN: use radius in rotation factor output, default=1
 * 
 * @return OUT: error code
 */
CVAPI(CVStatus) cvShapeFitting(
    const CvArr * shape0,
    const CvArr * shape1,
    double & scale,
    double & theta,
    double & tx, double & ty,
    const bool use_radius CV_DEFAULT(1),
    const CvMat * weights = NULL // CV_32F, weights of each point on curve
                              );

/**
 * Apply transformation on shape, with in-place modification
 * 
 * new_shape = [s*cos(theta) -s*sin(theta);
 *              s*sin(theta) +s*cos(theta)] * ...
 *             [old_shape.x; old_shape.y] + [tx; ty]
 * 
 * @param shape in: CV_32SC2, in-place transformed by the input factors
 * @param scale in: scale factor 
 * @param theta in: rotation factor
 * @param tx    in: translation factor on X-coord
 * @param ty    in: translation factor on Y-coord
 * 
 * @return error code
 */
CVAPI(CVStatus) cvShapeTransform(
    CvArr * shape,          // in/out: inplace transform the original shape
    const double scale,     // in: scale factor 
    const double theta,     // in: rotation factor
    const double tx,        // in: translation factor on X-coord
    const double ty         // in: translation factor on Y-coord
                          );

// affine warp
void cvShapeTransform2(
    CvMat * shape,
    const double scale,     // in: scale factor 
    const double theta,     // in: rotation factor
    const double tx,        // in: translation factor on X-coord
    const double ty         // in: translation factor on Y-coord
					   );


enum {CV_ALIGN_FA,CV_ALIGN_IC};
CVAPI(CVStatus) cvAlignImage_Affine(const CvArr * _image, 
									const CvArr * _tmplt, 
									CvArr * _init_p,
                                    int method=CV_ALIGN_FA);

/** 
 * 
 * 
 * @param src IN: CV_32FC1 supported ONLY, single row with multiple columns
 * @param dst OUT: same as input
 * @param idx OUT: CV_32SC1
 */
#if CV_MAJOR_VERSION!=2 && !defined(ANDROID) && !defined(CV_SORT_DESCENDING)
#define CV_SORT_ASCE  0
#define CV_SORT_DESC  1
CVAPI(void) cvSort(const CvArr * src, CvArr * dst,
                   CvArr * idx CV_DEFAULT(NULL),
                   const int order CV_DEFAULT(CV_SORT_ASCE) );
#endif // define cvSort for cxcore100
// cvMeanShiftEx
//
// @param src     	input data points, NxD matrix with FLOAT32 data (CV_32F data type assumed)
// @param dst     	mean shifted data points, same type and size as input
// @param ksize   	kernel size of Gaussian or flat kernel ( currently ONLY flat kernel supported)
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
//	IplImage * dispImage = cvCreateImage(cvSize(320,240),IPL_DEPTH_8U,3);cvZero(dispImage);
//	for (int i = 0; i < npoints; i++)
//		cvCircle(dispImage, cvPoint(CV_MAT_ELEM(orig,float,i,0),CV_MAT_ELEM(orig,float,i,1)),1,cvScalar(255),-1);
//	for (int i = 0; i < npoints; i++)
//		cvCircle(dispImage, cvPoint(CV_MAT_ELEM(*data,float,i,0),CV_MAT_ELEM(*data,float,i,1)),1,cvScalar(0,255),-1);
//	cvShowImage("Test", dispImage); cvWaitKey();
//	cvReleaseMat(&data);
//	cvReleaseImage(&dispImage);
//	cvDestroyWindow("Test");
// 
void cvMeanShiftEx(const CvArr * _src, CvArr * _dst, 
				   const float ksize=2.0f,
				   CvTermCriteria criteria=cvTermCriteria(3,10,0.1));

CV_INLINE
void icvBoundary(CvMat * bw)
{
  assert(CV_MAT_TYPE(bw->type)==CV_8U);
  uchar * ptr=bw->data.ptr;
  int j,k,step=bw->step/sizeof(uchar);
  int nr=bw->rows,nc=bw->cols;
  for (j=0;j<nr-1;j++,ptr+=step)
    for (k=0;k<nc-1;k++)
    {
      ptr[k]=((ptr[k+1]!=ptr[k]) || ((ptr+step)[k]!=ptr[k])) ? 255:0;
    }    
}

CV_INLINE
CvMat * icvRawRead(char * fn)
{
  int tmpval;
  FILE * fp = fopen(fn, "r");
  if (!fp) { fprintf(stderr, "ERROR: can't load %s !!\n", fn); return 0; }
  else {fprintf(stderr, "INFO: file %s loaded !!\n", fn);}
  int nr,nc,i,j;
  tmpval=fscanf(fp, "%d %d\n", &nr, &nc);
  CvMat * mat = cvCreateMat(nr, nc, CV_32F);
  int step=mat->step/sizeof(float);
  float * fptr=mat->data.fl;
  for (i=0;i<nr;i++) {
    for (j=0;j<nc;j++) {
      tmpval=fscanf(fp, "%f ", &fptr[j]);
    }
    fptr+=step;
  }
  fclose(fp);
  return mat;
}

CV_INLINE
int icvGaussianKernel(CvMat * X, CvMat * X_test, CvMat * K)
{
  int N0 = X->rows;
  int N1 = X_test->rows;
  int M = X_test->cols;
  assert(X->cols==M);
  assert((K->rows==N1)&&(K->cols==N0)&&(CV_MAT_TYPE(K->type)==CV_32F));
  // for testing data
  CvMat * tmpN1xM = cvCreateMat(N1,M,CV_32F);
  CvMat * tmpN1x1 = cvCreateMat(N1,1,CV_32F);
  CvMat * tmpN1xN0 = cvCreateMat(N1,N0,CV_32F);
  cvMul(X_test,X_test,tmpN1xM);
  cvReduce(tmpN1xM,tmpN1x1,-1,CV_REDUCE_SUM);
  cvRepeat(tmpN1x1,tmpN1xN0);
  // for training data
  CvMat * tmpN0xM = cvCreateMat(N0,M,CV_32F);
  CvMat * tmpN0x1 = cvCreateMat(N0,1,CV_32F);
  CvMat * tmpN0xN1 = cvCreateMat(N0,N1,CV_32F);
  cvMul(X,X,tmpN0xM);
  cvReduce(tmpN0xM,tmpN0x1,-1,CV_REDUCE_SUM);
  cvRepeat(tmpN0x1,tmpN0xN1);

  cvGEMM(X_test,X,-2,tmpN0xN1,1,K,CV_GEMM_B_T+CV_GEMM_C_T);
  // cvAdd(K,tmpN1xN0,K);
  // cvScale(K,K,-.5);
  // cvExp(K,K);
  {
    float * kptr = K->data.fl;
    float * tptr = tmpN1xN0->data.fl;
    int i,j,kstep=K->step/sizeof(float),tstep=tmpN1xN0->step/sizeof(float);
    for (i=0;i<N1;i++,kptr+=kstep,tptr+=tstep){
      for (j=0;j<N0;j++){kptr[j] = exp(-.5*(kptr[j]+tptr[j]));}
    }
  }
  
  cvReleaseMat(&tmpN1xM);
  cvReleaseMat(&tmpN1x1);
  cvReleaseMat(&tmpN1xN0);
  cvReleaseMat(&tmpN0xM);
  cvReleaseMat(&tmpN0x1);
  cvReleaseMat(&tmpN0xN1);
  return 1;
}

/** 
 * 
 * 
 * @param boxes      N0x4 int-type matrix (denotes top-left and
 *                   bottom-right corner of the rectangle) as input 
 * @param top        N1x4 int-type matrix as output
 * @param overlap    overlap ratio
 * 
 * @return 
 */
CV_INLINE
int icvNonMaxSuppress(CvMat * boxes, CvMat * top, const float overlap=.5)
{
  int N0 = boxes->rows, N1=0;
  assert((boxes->cols==4)&&(top->cols==4));
  if (N0==0){return 0;}
  CvMat * area = cvCreateMat(N0,1,CV_32S);
  
  cvReleaseMat(&area);
  return N1;
}

// CVAPI(void) cvSigmoid(CvMat * src, CvMat * dst,
//   int islogit = 0, float a = 1.7195, float b = 0.6666667);
// CVAPI(void) cvSigmoidDrv(CvMat * src, CvMat * dst,
//   int islogit = 0, float a = 1.7195, float b = 0.6666667);

#endif //__CV_EXT_C_H__
