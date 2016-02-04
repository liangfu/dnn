/**
* @file   cvext_pca.hpp
* @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
* @date   Wed Nov 28 09:21:18 2012
* 
* @brief  interface of classes based on PCA
* 
* 
*/

#include "cvaam.h"

/**
 * NxM matrix as input data
 * on each rows, X = [x0,y0,x1,y1,... x_k,x_k];
 * so that N = ${number-of-columns}
 */
void CvActiveShapeModel::train(const CvMat * shapelist)
{
  clear();
  CvMat * mean = cvCreateMat(1, shapelist->cols, CV_64F);
  {
    int M = shapelist->cols;
    int N = shapelist->rows;
    double sum = 0.0;
    for (int i = 0; i < M; i++){
      sum = 0.0;
      for (int j = 0; j < N; j++){
        sum += cvmGet(shapelist, j, i);
      }
      cvmSet(mean, 0, i, sum/N);
    }

    for (int i = 0; i < shapelist->rows; i++){
      CvMat subshapehdr, hdr2;
      CvMat * subshape =
          cvGetSubRect(shapelist, &subshapehdr,
                       cvRect(0,i,shapelist->cols,1));
      double scale, theta, tx, ty;
      fit_shape(mean, subshape, scale, theta, tx, ty);
      transform_shape(subshape, scale, theta, tx, ty);
      //transform_shape(subshape, 1, 0, 0, 0);
    }

    if (0)
    {
      IplImage * dispImage =
          cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
      cvZero(dispImage);
      for (int i = 0; i < 23; i++){
        cvCircle(dispImage, cvPoint(cvmGet(mean, 0, i*2+0),
                                    cvmGet(mean, 0, i*2+1)),
                 1, cvScalarAll(255),-1);
      }
      for (int j = 0; j < 12; j++)
        for (int i = 0; i < 23; i++){
          cvCircle(dispImage, cvPoint(cvmGet(shapelist, j, i*2+0),
                                      cvmGet(shapelist, j, i*2+1)),
                   1, cvScalar(0,0,255),-1);
        }
      cvShowImage("Test", dispImage); CV_WAIT();
      cvReleaseImageEx(dispImage);
    }

    pca.set_data(shapelist);
  }
  cvReleaseMatEx(mean);

  const int npoints = pca.mean->cols/2;
  m_meanshape = cvCreateMat(2, npoints, CV_64F);
  for (int i = 0; i < npoints; i++){
    cvmSet(m_meanshape, 0, i, cvmGet(pca.mean, 0, i*2+0));
    cvmSet(m_meanshape, 1, i, cvmGet(pca.mean, 0, i*2+1));
  }
}

/** 
 * fit a shape into a given mean model for shape alignment
 * 
 * @param _mean   IN: 1x2N matrix for mean shape representation
 * @param _shape  IN: 1x2N matrix
 * @param scale   out: 
 * @param theta   out: 
 * @param tx      out: 
 * @param ty      out: 
 */
void CvActiveShapeModel::fit_shape(
	const CvArr * _mean,  // 1x2N
	const CvArr * _shape, // 1x2N
	double & scale, double & theta,
	double & tx, double & ty)
{
  CvMat shapeheader, meanheader;
  CvMat * mean, * shape;
  if (CV_IS_MAT(_mean)){mean =  (CvMat*)_mean;}else{
    mean=cvGetMat(&_mean, &meanheader);}
  if (CV_IS_MAT(_shape)){shape=(CvMat*)_shape;}else{
    shape = cvGetMat(&_shape, &shapeheader);}

  assert(shape->rows==1 && mean->rows==1);
  // sequence of x0,y0,x1,y1, ... x_m, y_m;
  assert(CV_MAT_TYPE(mean->type)==CV_64F);
  assert(CV_MAT_TYPE(shape->type)==CV_64F);// same as above

  const int ncols = shape->cols;
  const int npoints = ncols/2;

  double
      X1=0.0, X2=0.0, Y1=0.0, Y2=0.0,
      C1=0.0, C2=0.0, W=0.0,  Z=0.0;
  double ax=0.0, ay=0.0;
  tx=0.0, ty=0.0;
  scale=1.0, theta=0.0;

  float * wvector = new float[npoints];
  for (int i = 0; i < npoints; i++) {wvector[i] = 1;}
  CvPoint2D64f * refpts = new CvPoint2D64f[npoints];
  CvPoint2D64f * newpts = new CvPoint2D64f[npoints];
  memcpy(refpts, mean->data.ptr, sizeof(CvPoint2D64f)*npoints);
  memcpy(newpts, shape->data.ptr, sizeof(CvPoint2D64f)*npoints);

  CvMat *A, *b, *x, *A_invert;
  A = cvCreateMat(4,4,CV_64FC1);
  A_invert = cvCreateMat(4,4,CV_64FC1);
  b = cvCreateMat(4,1,CV_64FC1);
  x = cvCreateMat(4,1,CV_64FC1);

  cvZero(A);
  cvZero(A_invert);
  cvZero(b);
  cvZero(x);

  //obtain the matrix X1, Y1, W
  for(int i = 0; i < npoints; i++)
  {
    X1 = X1 + wvector[i]*(refpts[i].x);
    Y1 = Y1 + wvector[i]*(refpts[i].y);
    W = W + wvector[i];
  }

  //obtain the matrix X2, Y2, Z

  X2=0.0; Y2=0.0; C1=0.0; C2=0.0; Z=0.0;

  for (int j = 0; j < npoints; j++)
  {
    X2 = X2 + wvector[j]*newpts[j].x;
    Y2 = Y2 + wvector[j]*newpts[j].y;
    C1 = C1 + wvector[j]*(refpts[j].x * newpts[j].x +
                          refpts[j].y * newpts[j].y);
    C2 = C2 + wvector[j]*(refpts[j].y * newpts[j].x -
                          refpts[j].x * newpts[j].y);
    Z  = Z  + wvector[j]*(newpts[j].x * newpts[j].x +
                          newpts[j].y * newpts[j].y);
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

  if (cvInvert( A, A_invert, CV_LU )==0.0){
    cvInvert( A, A_invert, CV_SVD );
  }

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
  delete [] refpts;
  delete [] newpts;
  cvReleaseMatEx(A);
  cvReleaseMatEx(A_invert);
  cvReleaseMatEx(x);
  cvReleaseMatEx(b);
}

void CvActiveShapeModel::transform_shape(CvMat * src, 
                                         const double scale,
                                         const double theta, 
                                         const double tx,
                                         const double ty)
{
  //cvShapeTransform2(src, scale, theta, tx, ty);
  CvMat hdr, hdr2;
  CvMat * shape0 = cvGetMat(src, &hdr); 
  assert(CV_MAT_TYPE(shape0->type)==CV_64F);
  assert(shape0->rows==1);
  const int npoints = shape0->cols/2;

  CvMat * shape = cvCreateMat(2,shape0->cols/2, CV_64F);
  for (int i = 0; i < npoints; i++){
    CV_MAT_ELEM(*shape, double, 0, i)=CV_MAT_ELEM(*shape0, double, 0, i*2);
    CV_MAT_ELEM(*shape, double, 1, i)=CV_MAT_ELEM(*shape0, double, 0, i*2+1);
  }

  CvMat * M = cvCreateMat(2,2,CV_64F);
  double cos_st = scale*cos(theta);
  double sin_st = scale*sin(theta);
  cvmSet(M, 0, 0, cos_st);
  cvmSet(M, 0, 1, -sin_st);
  cvmSet(M, 1, 0, sin_st);
  cvmSet(M, 1, 1, cos_st);

  //Xj = M*Xj+[tx ty]';
  CvMat * T = cvCreateMat(2,1,CV_64F);
  cvmSet(T, 0,0,tx);
  cvmSet(T, 1,0,ty);

  CvMat * T_rep = cvCreateMat(2,npoints,CV_64F);
  cvRepeat(T, T_rep);

  cvMatMulAdd(M,shape,T_rep,shape);

  for (int i = 0; i < npoints; i++){
    CV_MAT_ELEM(*shape0, double, 0, i*2)=CV_MAT_ELEM(*shape, double, 0, i);
    CV_MAT_ELEM(*shape0, double, 0, i*2+1)=CV_MAT_ELEM(*shape, double, 1, i);
  }

  cvReleaseMatEx(shape);
  cvReleaseMatEx(M);
  cvReleaseMatEx(T);
  cvReleaseMatEx(T_rep);
}

void CvActiveShapeModel::save(const char * datafn)
    // data file name - .txt file
{
  assert(pca.mean && pca.eigvec);
  FILE * fp = fopen(datafn, "w");
  fprintf(fp, "mean:\n");
  fprintf(fp, "%d %d\n", pca.mean->rows, pca.mean->cols);
  cvPrintEx(fp, pca.mean);
  fprintf(fp, "eigval:\n");
  fprintf(fp, "%d %d\n", pca.eigval->rows, pca.eigval->cols);
  cvPrintEx(fp, pca.eigval);
  fprintf(fp, "eigvec:\n");
  fprintf(fp, "%d %d\n", pca.eigvec->rows, pca.eigvec->cols);
  cvPrintEx(fp, pca.eigvec);
  fclose(fp);
}

void CvActiveShapeModel::load(const char * datafn) // data file name
{
  pca.clear();
  FILE * fp = fopen(datafn, "r");
  int nrows, ncols; char tmp[CV_MAXSTRLEN];
  float fval;

  fscanf(fp, "%s\n", tmp); 
  assert(strcmp(tmp,"mean:")==0);
  fscanf(fp, "%d %d", &nrows, &ncols);
  pca.mean = cvCreateMat(nrows, ncols, CV_64F);
  for (int i = 0; i < nrows; i++){
    for (int j = 0; j < ncols; j++){
      fscanf(fp, "%f", &fval);
      cvmSet(pca.mean, i, j, fval);
    }
  }

  fscanf(fp, "%s\n", tmp); 
  assert(strcmp(tmp,"eigval:")==0);
  fscanf(fp, "%d %d", &nrows, &ncols);
  pca.eigval = cvCreateMat(nrows, ncols, CV_64F);
  for (int i = 0; i < nrows; i++){
    for (int j = 0; j < ncols; j++){
      fscanf(fp, "%f", &fval);
      cvmSet(pca.eigval, i, j, fval);
    }
  }

  fscanf(fp, "%s\n", tmp);
  assert(strcmp(tmp,"eigvec:")==0);
  fscanf(fp, "%d %d", &nrows, &ncols);
  pca.eigvec = cvCreateMat(nrows, ncols, CV_64F);
  for (int i = 0; i < nrows; i++){
    for (int j = 0; j < ncols; j++){
      fscanf(fp, "%f", &fval);
      cvmSet(pca.eigvec, i, j, fval);
    }
  }

  fclose(fp);
}

void CvActiveShapeModel::fit(// prior shape
							 CvArr * _shape,
							 // grayscale image of current frame
							 const CvArr * _curr,
							 // grayscale image of next frame
							 const CvArr * _next,
							 // gradient image - precomputed 
							 const CvArr * _grad, 
							 // difference with next frame in time sequence
							 const CvArr * _diff )
{
  const int accuracy_param = 8;

  CvMat shapehdr; IplImage currhdr,nexthdr;
  CvMat * shape = cvGetMat(_shape, &shapehdr);
  IplImage * curr = cvGetImage(_curr, &currhdr);
  IplImage * next = cvGetImage(_next, &nexthdr);
  const int npoints = shape->cols/2;
static int iii=0;

  // deform according to image feature
  //if (0)
  { 
    deform(shape, curr, next);
    if (iii++%2!=0)return;
  }

  double scale, theta, tx, ty;
  double scale2, theta2, tx2, ty2;
  fit_shape(pca.mean, shape, scale, theta, tx, ty);
  fit_shape(shape, pca.mean, scale2, theta2, tx2, ty2);
  transform_shape(shape, scale, theta, tx, ty);

  // double scale, theta, tx, ty;
  // fit_shape(m_meanshape, _shape, scale, theta, tx, ty);
  // transform_shape(_shape, scale, theta, tx, ty);

  CvMat * dx = cvCreateMat(1, npoints*2, CV_64F);
  cvSub(shape, pca.mean, dx);

  // init P_inv if not initialized
  if (m_P_inv==NULL){
    m_P_inv = cvCreateMat(pca.eigvec->rows,pca.eigvec->cols,CV_64F);
    if (cvInvert(pca.eigvec,m_P_inv,CV_LU)==0.0)
      cvInvert(pca.eigvec,m_P_inv,CV_SVD); // time consuming - approx. 2ms
  }

  CvMat * db = cvCreateMat(m_P_inv->rows,1,CV_64F);
  {
    CvMat * dx_t = cvCreateMat(dx->cols, 1, CV_64F);
    cvTranspose(dx, dx_t);

    cvMatMul(m_P_inv, dx_t, db);

    cvReleaseMatEx(dx_t);
  }
  CvMat * b = cvCreateMat(m_P_inv->rows,1,CV_64F); 
  cvZero(b);

  // b = b+db;
  cvAdd(b, db, b);
  //cvAdd(pca.eigval, db, b);

  // limit range of b: -3*sqrt(lamda) < b_k < 3*sqrt(lamda)
  double lamda = sqrt(cvmGet(pca.eigval,accuracy_param,0))*2.;
  double b_limit[2] = {-lamda, lamda};
  for (int i = 0; i < b->rows; i++){
    double dval = b->data.db[i];
    //cvmGet(b, i, 0);
    if (dval<b_limit[0]) {b->data.db[i]=b_limit[0];}
    //{cvmSet(b, i, 0,b_limit[0]);}
    else if (dval>b_limit[1]) {b->data.db[i]=b_limit[1];}
    //{cvmSet(b, i, 0,b_limit[1]);}
  }

  CvMat * Pb = cvCreateMat(pca.eigvec->rows,1,CV_64F);
  cvMatMul(pca.eigvec, b, Pb);

  {
    CvMat * tmpmean = cvCloneMat(pca.mean);
    CvMat * Pb_t = cvCreateMat(Pb->cols, Pb->rows, CV_64F);
    cvTranspose(Pb, Pb_t);

    transform_shape(tmpmean, scale2, theta2, tx2, ty2
                    /*scale, theta, tx, ty*/);

    // new_shape = new_mean + Pb
    cvAdd(tmpmean, Pb_t, shape);
    //cvCopy(tmpmean, shape);

    cvReleaseMatEx(Pb_t);
    cvReleaseMatEx(tmpmean);
  }

  // display - comparison between mean shape and constructed shape
  if (0)
  {
    IplImage * dispImage = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    if (1){cvZero(dispImage);}else{cvCopy(curr, dispImage);}

    CvPoint pt1,pt2;
    for (int i = 0; i < npoints*2; i+=2){
      pt1 = cvPoint(cvmGet(pca.mean, 0, i), cvmGet(pca.mean, 0, i+1));
      pt2 = cvPoint(cvmGet(shape, 0, i), cvmGet(shape, 0, i+1));
      cvCircle(dispImage, pt1, 1, cvScalarAll(255), -1);
      cvCircle(dispImage, pt2, 1, cvScalar(0,0,255), -1);
    }
    cvShowImage("Test", dispImage); CV_WAIT();
    cvReleaseImageEx(dispImage);
  }

  cvReleaseMatEx(Pb);
  cvReleaseMatEx(db);
  cvReleaseMatEx(b);
  cvReleaseMatEx(dx);
}

void CvActiveShapeModel::deform(CvMat * shape,
                                const CvArr * curr,
                                const CvArr * next)
{
  assert(shape->rows==1);
  const int npoints = shape->cols/2;

  //////////////// motion information is available
  if (next!=NULL)
  {
    CvMat * pts = cvCreateMat(1, shape->cols/2, CV_32SC2);
    for (int i = 0; i < npoints; i++){
      ((CvPoint*)pts->data.ptr)[i] =
          cvPoint(shape->data.db[i*2], shape->data.db[i*2+1]);
    } //cvShowImage("Test", curr); CV_WAIT();
    cvOpticalFlowPointTrack(curr, next, pts/*CV_32SC2*/, cvSize(15,15),4);
    for (int i = 0; i < npoints; i++){
      shape->data.db[i*2]=((CvPoint*)pts->data.ptr)[i].x;
      shape->data.db[i*2+1]=((CvPoint*)pts->data.ptr)[i].y;
    } //cvShowImage("Test", curr); CV_WAIT();
    cvReleaseMatEx(pts);
  }

  //////////////// only local information is available
  if (0)
  {
    CvPoint2D32f * pts = new CvPoint2D32f[npoints];
    for (int i = 0; i < npoints; i++){
      pts[i] = cvPoint2D32f(shape->data.db[i*2], shape->data.db[i*2+1]);
    }
    cvFindCornerSubPix(curr, pts, npoints,
                       cvSize(3,3), cvSize(-1,-1),
                       cvTermCriteria(1,100,0.1));
    delete [] pts;
  }
}

////////////////////////////////////////////////////////////////
// ACTIVE APPEARANCE MODEL
////////////////////////////////////////////////////////////////

void cvAppearanceModelLearn(
    CvAppearanceModel * aam,
    const CvMat * _shape,
    const CvArr * _image)
{
  CvMat * shape;
  CvMat * image, imagestub;
  CV_FUNCNAME("cvAppearanceModelLearn");
  __BEGIN__;
  CV_ASSERT(CV_IS_MAT(_shape));

  shape = cvCloneMat(_shape);
  image = cvGetMat(_image, &imagestub);
  CV_ASSERT(shape->rows==1);
  
  double scale, theta, tx, ty;
  aam->fit_shape(aam->pca.mean, shape, scale, theta, tx, ty);

  //////////////// for texture collection
  {
    CvPoint2D32f srcpts[3], dstpts[3];
    // collect points before transform
    srcpts[0] = cvPoint2D32f(shape->data.db[8], shape->data.db[9]);
    srcpts[1] = cvPoint2D32f(shape->data.db[22], shape->data.db[23]);
    srcpts[2] = cvPoint2D32f(shape->data.db[36], shape->data.db[37]);

    // collect points after transform
    aam->transform_shape(shape, scale, theta, tx, ty);
    dstpts[0] = cvPoint2D32f(shape->data.db[8], shape->data.db[9]);
    dstpts[1] = cvPoint2D32f(shape->data.db[22], shape->data.db[23]);
    dstpts[2] = cvPoint2D32f(shape->data.db[36], shape->data.db[37]);

    double ax = scale*cos(theta); double ay = scale*sin(theta);
    double init_p_data[6] = {ax, -ay, tx, ay, ax, ty};

    CvMat init_p = cvMat(2,3,CV_64F, init_p_data);
    //cvGetAffineTransform(srcpts, dstpts, init_p);

    {
      IplImage * dispImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
      {
        cvWarpAffine(image, dispImage, &init_p);
        //for (int i = 0; i < npoints; i++){
        //  
        //}
      }
      cvDrawLandmarks(dispImage, shape);
      cvShowImage("Test", dispImage); CV_WAIT();
      cvReleaseImageEx(dispImage);
    }
  }
  
  cvReleaseMatEx(shape);

  __END__;
}

void cvAppearanceModelInference(
    CvAppearanceModel * aam,
    const CvMat * shape,
    CvArr * _curr ,
    CvArr * _next ,
    CvArr * _grad )
{}
