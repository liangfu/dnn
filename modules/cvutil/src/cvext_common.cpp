/**
 * @file   cvext.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Nov 09 11:23:22 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvext_c.h"

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
#define CV_EXT_SQR_DIST(x,y) ((int)(x) * (x) + (y) * (y))
/** 
 * Euclidean distance transform of a binary image,
 * Computation performed upon 'short' values
 * 
 * @param src       IN: positive pixels are treated as background
 * @param distxArr  OUT: 
 * @param distyArr  OUT: 
 */
void cvCalcEuclideanDistance(
    const CvArr * src,      // positive pixels are treated as background
    CvArr * distxArr, CvArr * distyArr)
{
  CV_FUNCNAME("cvEuclideanDistance");
  
  int x, y, i;
  int offset_u, offset_ur, offset_r, offset_rd,
      offset_d, offset_dl, offset_l, offset_lu;
  double olddist2, newdist2, newdistx, newdisty;
  int changed;

  __BEGIN__;
  // CV_ASSERT(cvGetElemType(src)==CV_16S);
  // CV_ASSERT(cvGetElemType(distxArr)==CV_16S);
  // CV_ASSERT(cvGetElemType(distyArr)==CV_16S);

  CvMat srcHeader, distHeaderX, distHeaderY;
  CvMat * srcImage = cvGetMat(src, &srcHeader);
  CvMat * distImageX = cvGetMat(distxArr, &distHeaderX);
  CvMat * distImageY = cvGetMat(distyArr, &distHeaderY);
  CvMat * srcImage2 =
      cvCreateMat(srcImage->rows, srcImage->cols, CV_16S);
  CvMat * distImageX2 =
      cvCreateMat(srcImage->rows, srcImage->cols, CV_16S);
  CvMat * distImageY2 =
      cvCreateMat(srcImage->rows, srcImage->cols, CV_16S);

  if (cvGetElemType(src)!=CV_16S) cvConvert(srcImage, srcImage2);
  else cvCopy(srcImage, srcImage2);

  CvMat distImageXheader, distImageYheader;
  CvMat * distImageXreshaped =
      cvReshape(distImageX2, &distImageXheader, 0, 1);
  CvMat * distImageYreshaped =
      cvReshape(distImageY2, &distImageYheader, 0, 1);
  
  int w = srcImage->cols;
  int h = srcImage->rows;
  short * distx = (short*)distImageXreshaped->data.ptr;
  short * disty = (short*)distImageYreshaped->data.ptr;

  // Initialize index offsets for the current image width
  offset_u = -w;
  offset_ur = -w+1;
  offset_r = 1;
  offset_rd = w+1;
  offset_d = w;
  offset_dl = w-1;
  offset_l = -1;
  offset_lu = -w-1;

  /* Initialize the distance images to be all large values */
  for(i=0; i<w*h; i++)
    // if(img(i) <= 0.0)
    if(CV_MAT_ELEM(*srcImage2, short, i/w, i%w) <= 0.0)
    {
      distx[i] = 32000; // Large but still representable in a short, and
      disty[i] = 32000; // 32000^2 + 32000^2 does not overflow an int
    }
    else
    {
      distx[i] = 0;
      disty[i] = 0;
    }

  /* Perform the transformation */
  do
  {
    changed = 0;

    /* Scan rows, except first row */
    for(y=1; y<h; y++)
    {

      /* move index to leftmost pixel of current row */
      i = y*w; 

      /* scan right, propagate distances from above & left */

      /* Leftmost pixel is special, has no left neighbors */
      olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
      if(olddist2 > 0) // If not already zero distance
      {
        newdistx = distx[i+offset_u];
        newdisty = disty[i+offset_u]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_ur]-1;
        newdisty = disty[i+offset_ur]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }
      i++;

      /* Middle pixels have all neighbors */
      for(x=1; x<w-1; x++, i++)
      {
        olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
        if(olddist2 == 0) continue; // Already zero distance

        newdistx = distx[i+offset_l]+1;
        newdisty = disty[i+offset_l];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_lu]+1;
        newdisty = disty[i+offset_lu]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_u];
        newdisty = disty[i+offset_u]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_ur]-1;
        newdisty = disty[i+offset_ur]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }

      /* Rightmost pixel of row is special, has no right neighbors */
      olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
      if(olddist2 > 0) // If not already zero distance
      {
        newdistx = distx[i+offset_l]+1;
        newdisty = disty[i+offset_l];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_lu]+1;
        newdisty = disty[i+offset_lu]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_u];
        newdisty = disty[i+offset_u]+1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }
      }

      /* Move index to second rightmost pixel of current row. */
      /* Rightmost pixel is skipped, it has no right neighbor. */
      i = y*w + w-2;

      /* scan left, propagate distance from right */
      for(x=w-2; x>=0; x--, i--)
      {
        olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
        if(olddist2 == 0) continue; // Already zero distance
              
        newdistx = distx[i+offset_r]-1;
        newdisty = disty[i+offset_r];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }
    }
      
    /* Scan rows in reverse order, except last row */
    for(y=h-2; y>=0; y--)
    {
      /* move index to rightmost pixel of current row */
      i = y*w + w-1;

      /* Scan left, propagate distances from below & right */

      /* Rightmost pixel is special, has no right neighbors */
      olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
      if(olddist2 > 0) // If not already zero distance
      {
        newdistx = distx[i+offset_d];
        newdisty = disty[i+offset_d]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_dl]+1;
        newdisty = disty[i+offset_dl]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }
      i--;

      /* Middle pixels have all neighbors */
      for(x=w-2; x>0; x--, i--)
      {
        olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
        if(olddist2 == 0) continue; // Already zero distance

        newdistx = distx[i+offset_r]-1;
        newdisty = disty[i+offset_r];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_rd]-1;
        newdisty = disty[i+offset_rd]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_d];
        newdisty = disty[i+offset_d]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_dl]+1;
        newdisty = disty[i+offset_dl]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }
      /* Leftmost pixel is special, has no left neighbors */
      olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
      if(olddist2 > 0) // If not already zero distance
      {
        newdistx = distx[i+offset_r]-1;
        newdisty = disty[i+offset_r];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_rd]-1;
        newdisty = disty[i+offset_rd]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }

        newdistx = distx[i+offset_d];
        newdisty = disty[i+offset_d]-1;
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          olddist2=newdist2;
          changed = 1;
        }
      }

      /* Move index to second leftmost pixel of current row. */
      /* Leftmost pixel is skipped, it has no left neighbor. */
      i = y*w + 1;
      for(x=1; x<w; x++, i++)
      {
        /* scan right, propagate distance from left */
        olddist2 = CV_EXT_SQR_DIST(distx[i], disty[i]);
        if(olddist2 == 0) continue; // Already zero distance

        newdistx = distx[i+offset_l]+1;
        newdisty = disty[i+offset_l];
        newdist2 = CV_EXT_SQR_DIST(newdistx, newdisty);
        if(newdist2 < olddist2)
        {
          distx[i]=newdistx;
          disty[i]=newdisty;
          changed = 1;
        }
      }
    }
  }
  while(changed); // Sweep until no more updates are made

  if (cvGetElemType(distxArr)!=CV_16S)
    cvConvert(distImageX2, distImageX);
  else cvCopy(distImageX2, distImageX);

  if (cvGetElemType(distyArr)!=CV_16S)
    cvConvert(distImageY2, distImageY);
  else cvCopy(distImageY2, distImageY);

  cvReleaseMatEx(srcImage2);
  cvReleaseMatEx(distImageX2);
  cvReleaseMatEx(distImageY2);
  /* The transformation is completed. */
  __END__;
}

/** 
 * perform point-polygon test for every point in given image frame
 * (Warning: this is very SLOW for large image!!!)
 * 
 * @param contour   IN:  1xN CV_32FC2, indicates contour
 * @param dst       OUT: signed distance function
 * 
 * @return OUT: status information
 */
// CVStatus cvCalcSignedDistance(const CvArr * src, CvArr * dst)
CVAPI(CVStatus) cvCalcDistTransform(const CvArr * src, CvArr * dst)
{
  CV_FUNCNAME("cvCalcDistTransform");
  int type = cvGetElemType(dst);
  CvSize imsize = cvGetSize(dst);
  CvMat header;
  CvMat * sdfImage, * retImage;
  __BEGIN__;
  CV_ASSERT(src&&dst);
  CV_ASSERT(type==CV_32F);
  retImage = cvGetMat(dst, &header);
  sdfImage = cvCreateMat(imsize.height, imsize.width, CV_32F);

  if (cvGetSize(src).height==1) // contour as input
  {
    IplImage * initImage =
        cvCreateImage(imsize, IPL_DEPTH_32F, 1);
    cvSet(initImage, cvScalarAll(-1));
    cvDrawContourEx(initImage, src, cvScalarAll(1), 1);
    IplImage * distx = cvCreateImage(imsize, IPL_DEPTH_32F, 1);
    IplImage * disty = cvCreateImage(imsize, IPL_DEPTH_32F, 1);
    cvCalcEuclideanDistance(initImage, distx, disty);
    cvCartToPolar(distx, disty, sdfImage);
    cvReleaseImageEx(distx);
    cvReleaseImageEx(disty);
    cvReleaseImageEx(initImage);
  }
  else if (cvGetSize(src).height!=1 &&
           cvGetSize(src).width!=1) // binary image as input
  {
    IplImage * distx = cvCreateImage(imsize, IPL_DEPTH_32F, 1);
    IplImage * disty = cvCreateImage(imsize, IPL_DEPTH_32F, 1);
    cvCalcEuclideanDistance(src, distx, disty);
    cvCartToPolar(distx, disty, sdfImage);
    cvReleaseImageEx(distx);
    cvReleaseImageEx(disty);
  }else{
    fprintf(stderr, "Warning: input matrix unrecognized !\n");
    assert(false);
  }

  if (type==CV_32F) // CV_16S might be supported as well, untested
  {
    cvCopy(sdfImage, retImage);
  }else{
    fprintf(stderr, "Warning: data type not supported !\n");
    assert(false);
  }

  cvReleaseMatEx(sdfImage);
  __END__;
  return CV_StsOk;
}

/** 
 * Align shapes, for getting mean shape from a list of shapes for training
 *
 * Reference:
 *     Active shape models - their training and applications, Appendix-A
 * 
 * @param shape0        IN: 1st shape for comparison, meanshape
 * @param shape1        IN: 2nd shape for comparison, new shape for training
 * @param scale         OUT: scale factor from 1st shape to 2nd
 * @param theta         OUT: rotation factor
 * @param tx            OUT: x-coord translation factor
 * @param ty            OUT: y-coord translation factor
 * @param use_radius    IN: use radius in rotation factor output, default=1
 * 
 * @return OUT: error code
 */
CVStatus cvShapeFitting(
    const CvArr * shape0, // CV_32SC2 representing the mean shape
    const CvArr * shape1, // CV_32SC2
    double & scale,
    double & theta,
    double & tx, double & ty,
    const bool use_radius,
    const CvMat * weights // CV_32F, weights of each point on curve
                       )
{
  CV_FUNCNAME("cvShapeFitting");
  CvMat shape0header, shape1header;

  CvMat * cc0 = cvGetMat(shape0, &shape0header);
  CvMat * cc1 = cvGetMat(shape1, &shape1header);
  CvMat * contour0 = cvCreateMat(cc0->rows, cc0->cols, CV_64FC2);
  CvMat * contour1 = cvCreateMat(cc0->rows, cc0->cols, CV_64FC2);
  CvPoint2D64f * meanshape;
  CvPoint2D64f * shape;
  CvMat *A, *b, *x, *A_invert;
  double
      X1=0.0, X2=0.0, Y1=0.0, Y2=0.0,
      C1=0.0, C2=0.0, W=0.0,  Z=0.0;
  double ax=0.0, ay=0.0;
  double determinant=0.0;
  float * wvector;
  __BEGIN__;
  if (cvGetElemType(cc0)==CV_32SC2 || cvGetElemType(cc0)==CV_32FC2){
    cvConvert(cc0, contour0);
  }else if (cvGetElemType(cc0)==CV_64FC2){
    cvCopy(cc0, contour0);
  }else{
    fprintf(stderr, "Error: data type not supported.\n");
#if CV_MAJOR_VERSION==1
    EXIT;
#endif
  }
  if (cvGetElemType(cc1)==CV_32SC2 || cvGetElemType(cc1)==CV_32FC2){
    cvConvert(cc1, contour1);
  }else if (cvGetElemType(cc1)==CV_64FC2){
    cvCopy(cc1, contour1);
  }else{
    fprintf(stderr, "Error: data type not supported.\n");
#if CV_MAJOR_VERSION==1
    EXIT;
#endif
  }

  CV_ASSERT(shape0&&shape1);
  //CV_ASSERT(cvGetElemType(shape0)==cvGetElemType(shape1));
  if (weights!=NULL){CV_ASSERT(cvGetElemType(weights)==CV_32F);}
  CV_ASSERT(contour0->cols==contour1->cols);
  CV_ASSERT(contour0->rows==1);
  CV_ASSERT(contour1->rows==1);

  tx=0.0, ty=0.0;
  scale=1.0, theta=0.0;

  wvector = new float[contour0->cols];
  if (weights!=NULL){
    memcpy(wvector, weights->data.ptr, sizeof(float)*contour0->cols);
  }else{
    for (int i = 0; i < contour0->cols; i++) wvector[i] = 1;
  }
  meanshape = new CvPoint2D64f[contour0->cols];
  shape     = new CvPoint2D64f[contour1->cols];
  memcpy(meanshape, contour0->data.ptr,
         sizeof(CvPoint2D64f)*contour0->cols);
  memcpy(shape, contour1->data.ptr,
         sizeof(CvPoint2D64f)*contour1->cols);

  A = cvCreateMat(4,4,CV_64FC1);
  A_invert = cvCreateMat(4,4,CV_64FC1);
  b = cvCreateMat(4,1,CV_64FC1);
  x = cvCreateMat(4,1,CV_64FC1);

  cvZero(A);
  cvZero(A_invert);
  cvZero(b);
  cvZero(x);

  //obtain the matrix X1, Y1, W
  for(int i = 0; i < contour0->cols; i++)
  {
    X1 = X1 + wvector[i]*(meanshape[i].x);
    Y1 = Y1 + wvector[i]*(meanshape[i].y);
    W = W + wvector[i];
  }

  //obtain the matrix X2, Y2, Z
  
  X2=0.0; Y2=0.0; C1=0.0; C2=0.0; Z=0.0;

  for (int j = 0; j < contour0->cols; j++)
  {
    X2 = X2 + wvector[j]*shape[j].x;
    Y2 = Y2 + wvector[j]*shape[j].y;
    C1 = C1 + wvector[j]*(meanshape[j].x * shape[j].x +
                         meanshape[j].y * shape[j].y);
    C2 = C2 + wvector[j]*(meanshape[j].y * shape[j].x -
                         meanshape[j].x * shape[j].y);
    Z  = Z  + wvector[j]*(shape[j].x * shape[j].x +
                         shape[j].y * shape[j].y);
  }

  //---------------------------
  //	A=[X2 -Y2 W   0
  //	   Y2  X2 0   W
  //	   Z   0  X2  Y2
  //	   0   Z  -Y2 X2]
  //---------------------------
  for(int k=0; k<4; ++k)
    cvmSet( A, k, k, X2);

  cvmSet( A, 0, 1, -Y2);
  cvmSet( A, 0, 2, W);

  cvmSet( A, 1, 0, Y2);
  cvmSet( A, 1, 3, W);

  cvmSet( A, 2, 0, Z);
  cvmSet( A, 2, 3, Y2);

  cvmSet( A, 3, 1, Z);
  cvmSet( A, 3, 2, -Y2);

  //------------------------------------------
  //	C = transpose of [ X1, Y1, C1, C2 ]
  //------------------------------------------
  cvmSet( b, 0, 0, X1);
  cvmSet( b, 1, 0, Y1);
  cvmSet( b, 2, 0, C1);
  cvmSet( b, 3, 0, C2);

// #if 1
//   determinant = cvInvert( A, A_invert, CV_LU );
//   if(determinant==0.0) // singular case
// #endif
  determinant = cvInvert( A, A_invert, CV_SVD );

  // A*x=b      -------------    x=inv(A)*b
  cvMatMul( A_invert, b , x );

  //solving the M matrix
  ax = cvmGet(x,0,0);
  ay = cvmGet(x,1,0);
  tx = cvmGet(x,2,0);
  ty = cvmGet(x,3,0);

  theta = atan(ay/ax);
  scale = ax/cos(theta);

  delete [] wvector;
  delete [] meanshape;
  delete [] shape;
  cvReleaseMatEx(A);
  cvReleaseMatEx(A_invert);
  cvReleaseMatEx(x);
  cvReleaseMatEx(b);

  __END__;
  cvReleaseMatEx(contour0);
  cvReleaseMatEx(contour1);
  return CV_StsOk;
}

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
CVStatus cvShapeTransform(
    CvArr * shape,          // in/out: transform the original shape
    const double scale,     // in: scale factor 
    const double theta,     // in: rotation factor
    const double tx,        // in: translation factor on X-coord
    const double ty         // in: translation factor on Y-coord
                          )
{
  CV_FUNCNAME("cvShapeTransform");
  // Xi = M(scale, theta)[Xj]+tj
  CvMat shapeheader, shapeReshapeHdr, *Xi, *Xj;
  
  __BEGIN__;
  Xi = cvGetMat(shape, &shapeheader);

  CvMat * M = cvCreateMat(2,2,CV_32F);
  double cos_tt = scale*cos(theta);
  double sin_tt = scale*sin(theta);
  cvmSet(M, 0, 0, cos_tt);
  cvmSet(M, 0, 1, -sin_tt);
  cvmSet(M, 1, 0, sin_tt);
  cvmSet(M, 1, 1, cos_tt);

  // scale, rotate and translate:
  // Xj = M*Xj+[tx ty]';
  cvTranspose(Xi, Xi);
  Xj = cvReshape(Xi, &shapeReshapeHdr, 1);
  CvMat * Xj_t = 
	  cvCreateMat(Xj->cols, Xj->rows, CV_MAT_TYPE(Xj->type));
  cvTranspose(Xj, Xj_t);
  if (cvGetElemType(Xj)!=CV_32F){
    CvMat * Xj32f = cvCreateMat(Xj_t->rows, Xj_t->cols, CV_32F);
    cvConvert(Xj_t, Xj32f);
    cvMatMul(M, Xj32f, Xj32f);
    for (int i = 0; i < Xj->cols; i++) {
      cvmSet(Xj32f, 0, i, cvmGet(Xj32f,0,i)+tx);
      cvmSet(Xj32f, 1, i, cvmGet(Xj32f,1,i)+ty);
    }
    cvConvert(Xj32f, Xj_t);
    cvReleaseMatEx(Xj32f);
  }else{
    cvMatMul(M, Xj_t, Xj_t);
    for (int i = 0; i < Xj->cols; i++) {
      cvmSet(Xj_t, 0, i, cvmGet(Xj_t,0,i)+tx);
      cvmSet(Xj_t, 1, i, cvmGet(Xj_t,1,i)+ty);
    }
  }
  cvTranspose(Xj_t, Xj);
  cvTranspose(Xi, Xi);
  cvReleaseMatEx(M);

  __END__;
  return CV_StsOk;
}

void cvShapeTransform2(
					   CvMat * src,
    const double scale,     // in: scale factor 
    const double theta,     // in: rotation factor
    const double tx,        // in: translation factor on X-coord
    const double ty         // in: translation factor on Y-coord
					   )
{
	CV_FUNCNAME("cvShapeTransform2");
    CvMat * shape;CvMat * M;CvMat * T;CvMat * T_rep;
    double cos_tt, sin_tt;

	__BEGIN__;
	CV_ASSERT(src->rows==2);
	shape = cvCreateMat(2, src->cols, CV_64F);
	cvConvert(src,shape);

	M = cvCreateMat(2,2,CV_64F);
	cos_tt = scale*cos(theta);
	sin_tt = scale*sin(theta);
	cvmSet(M, 0, 0, cos_tt);
	cvmSet(M, 0, 1, -sin_tt);
	cvmSet(M, 1, 0, sin_tt);
	cvmSet(M, 1, 1, cos_tt);

	//Xj = M*Xj+[tx ty]';
	T = cvCreateMat(2,1,CV_64F);
	cvmSet(T, 0,0,tx);
	cvmSet(T, 1,0,ty);

	T_rep = cvCreateMat(2,shape->cols,CV_64F);
	cvRepeat(T, T_rep);

	cvMatMulAdd(M,shape,T_rep,shape);

	cvConvert(shape, src);
	cvReleaseMatEx(shape);
	cvReleaseMatEx(M);
	cvReleaseMatEx(T);
	cvReleaseMatEx(T_rep);
	__END__;
}

/**
 * An example of polynomial function
 * float example_func(float a, float b, float c) {return a*x*x+b*x+c;}
 * for
 *     y = a*x*x+b*x+c;
 *
 * @param arr IN: row vector as a set of parameters
 * @param x   IN: sample point
 * 
 * @return OUT: value at sample point
 */
double polynomial_func(
    const CvArr * arr,        // (CV_32F||CV_64F) a set of parameters
    const double x            // sample point
                      )
{
  CV_FUNCNAME("polynomial_func");
  double retval = 0.0;
  CvMat * mat;
  __BEGIN__;
  CV_ASSERT(cvGetElemType(arr)==CV_32F || cvGetElemType(arr)==CV_64F);
  CvMat arrheader;
  mat = cvGetMat(arr, &arrheader);

  for (int i = 0; i < mat->cols; i++)
    retval += cvmGet(mat, 0, i)*pow(x, mat->cols-1-i);
  
  __END__;
  return retval;
}

enum {CV_FIT_LSQ=1};

/** 
 * least-square curve fitting
 * 
 * @param define_func   in:  defined function for curve fitting
 * @param samples       in:  a list of sample points
 * @param result        OUT: curve fitting results
 * @param method        in:  approach of the fitting method
 * 
 * @return OUT: error code
 */
CVStatus cvCurveFittingLS(
    double (*defined_func)(const CvArr*, const double),     // polynomial_func(arr)
    const CvArr * samples,                      // 
    CvArr * result,                             // CV_32F, a list of resulting parameters
    int method CV_DEFAULT(CV_FIT_LSQ)           // curve fitting method
                          )
{
  CV_FUNCNAME("cvCurveFittingLS");
  __BEGIN__;
  if (defined_func==NULL) {defined_func = &polynomial_func;}

  __END__;
  return CV_StsOk;
}

