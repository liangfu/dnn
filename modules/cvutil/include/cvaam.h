/**
 * @file   cvext_pca.hpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Wed Nov 28 09:21:18 2012
 * 
 * @brief  interface of classes based on PCA
 * 
 * 
 */

#ifndef __CV_EXT_PCA_H__
#define __CV_EXT_PCA_H__

#include "cvext_c.h"

/*
clear; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Principle Component Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 10; M = 2;

% generate data
X = randn(N, M);

% show data
if 0, plot(X(:,1),X(:,2),'.'); end

% mean substraction
mu = mean(X,1);
X_c = X-repmat(mu, [N 1]);

% eigenvector of covariance matrix
covX = cov(X);
[V D]=eig(covX');
D=diag(D);

% sort by eigenvalues
[junk idx] = sort(-1*D);
D = D(idx); % eigenvalues
V = V(:,idx); % eigenvector - principle components

% accuracy
rate = cumsum(D)/sum(D)

% what proportion are projected
if 1, V = V(:,1:3); end

% projection
proj = X_c*V;  % PRINCIPLE COMPONENT COEFF

% back projection
orig = proj*V'+repmat(mu, [N 1]);
X; % original data

% comparison with original data
if 1,
plot(orig(:,1),orig(:,2),'r.', ...
	 X(:,1),X(:,2),'b.');
end
*/
class CvPrincipleComponentAnalysis
{
  int M; // ncols;
  int N; // nrows;
  CvMat * X;   // original data: NxD, we reduce D-dim to M-dim
  CvMat * X_c; // mean subtracted
  CvMat * mean_rep;
  CvMat * covar;
  CvMat * proj; // projection: proj = eigvec'*X_c;
  CvMat * orig; // backprojected data - close to original data

  CvMat * rate; // accumulated percentage of eigenvalues

 public:
  CvMat * mean;   // mean = mean(X); % mean of original data
  CvMat * eigval; // as latent variables
  CvMat * eigvec; // [eigvec eigval]=eig(1/N*X_c*X_c');
  
 public:
  CvPrincipleComponentAnalysis():
	  M(0),N(0),
	  X(NULL),X_c(NULL),proj(NULL),orig(NULL),rate(NULL),mean(NULL),
	  eigvec(NULL),eigval(NULL)
  {}
  ~CvPrincipleComponentAnalysis(){clear();}
  
  void clear(){
    cvReleaseMatEx(X);
    cvReleaseMatEx(X_c);
    cvReleaseMatEx(proj);
    cvReleaseMatEx(orig);
	cvReleaseMatEx(rate);

    cvReleaseMatEx(mean);
    cvReleaseMatEx(eigvec);
    cvReleaseMatEx(eigval);
  }

  void set_data(const CvMat * mat, const bool use_proj = 1){
    clear();
    M = mat->cols;
    N = mat->rows;
    X = cvCreateMat(N, M, CV_64F);//cvCloneMat(mat);
    if (CV_MAT_TYPE(mat->type)==CV_64F){cvCopy(mat, X);}else{cvConvert(mat, X);}
    int minMN = MIN(M,N);

    mean = cvCreateMat(1, M, CV_64F);
    mean_rep = cvCreateMat(N, M, CV_64F);
    X_c = cvCreateMat(N, M, CV_64F);
#define xxx 0
#if xxx
	covar = cvCreateMat(N, N, CV_64F);
    eigvec = cvCreateMat(M, minMN, CV_64F);     // coefficients
    eigval = cvCreateMat(minMN, 1, CV_64F); // latent variables
    proj = cvCreateMat(N, minMN, CV_64F);
#else
	covar = cvCreateMat(M, M, CV_64F);
    eigvec = cvCreateMat(M, M, CV_64F);     // coefficients
    eigval = cvCreateMat(M, 1, CV_64F); // latent variables
    proj = cvCreateMat(N, M, CV_64F);
#endif

    calc_mean();
    cvRepeat(mean, mean_rep);  // repmat(mean, [N 1])
    cvSub(X,mean_rep,X_c);     // mean subtract
    // calculate covariance matrix
#if xxx
    cvGEMM(X_c,X_c,1/N,NULL,1.0f,covar, CV_GEMM_B_T);
    CvMat * eigvec_t = cvCreateMat(eigvec->cols, eigvec->rows, CV_64F);   // coefficients
    // [eigvec eigval]=eig(1/N*(data-mean)*(data-mean)');
    cvCalcPCA(X,mean,eigval,eigvec_t,
              CV_PCA_DATA_AS_ROW+CV_PCA_USE_AVG);
    cvTranspose(eigvec_t, eigvec);
    cvReleaseMatEx(eigvec_t);
#else
	//cvPrintEx(stderr, mean);puts("");
	//CvRect roi = cvRect(0,0,5,5);
	//cvPrintROI(stderr, mean_rep,roi);puts("");
	//cvPrintROI(stderr, X_c,roi);puts("");cvSave("X_c.xml",X_c);
    cvGEMM(X_c,X_c,double(1./N),NULL,1.,covar, CV_GEMM_A_T);
	//cvPrintROI(stderr, covar,roi);puts("");
	cvEigenVV(covar, eigvec, eigval, DBL_EPSILON);
	//cvSVD(covar, eigval, eigvec, 0, CV_SVD_U_T + CV_SVD_MODIFY_A);
#endif
#undef xxx
    //fprintf(stderr, "eigval:\n");
    //cvPrintEx(stderr, eigval);
    //fprintf(stderr, "\n");

    // projection : (X-mu)*eigvec
    if (use_proj) { project(); }
  }

  inline void project()
  {
    cvGEMM(X_c,eigvec,1.,NULL,1.,proj);
  }

  inline void back_project()
  {
    // orig = proj*eigvec'+mean_rep
    orig = cvCreateMat(N, M, CV_64F);
    cvGEMM(proj,eigvec,1.,mean_rep,1.,orig,CV_GEMM_B_T);
  }

  void calc_mean()
  {
    double sum = 0.0;
    for (int i = 0; i < M; i++){
      sum = 0.0;
      for (int j = 0; j < N; j++){
        sum += cvmGet(X, j, i);
      }
      cvmSet(mean, 0, i, sum/N);
    }
  }
};

class CvActiveShapeModel
{
 public:
  CvPrincipleComponentAnalysis pca;
  CvMat * m_meanshape;
  CvMat * m_P_inv;

 public:
  CvActiveShapeModel():
      m_meanshape(NULL),
		  m_P_inv(NULL)
  {}
  ~CvActiveShapeModel(){clear();}

  void clear()
  {
    cvReleaseMatEx(m_meanshape);
	cvReleaseMatEx(m_P_inv);
  }

  /**
   * NxM matrix as input data
   * on each rows, X = [x0,y0,x1,y1,... x_k,x_k];
   * so that N = ${number-of-columns}
   */
  virtual void train(const CvMat * shapelist);

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
  void fit_shape(
	  const CvArr * _mean,  // 1x2N
	  const CvArr * _shape, // 1x2N
	  double & scale, double & theta,
	  double & tx, double & ty);

  /**
   * affine shape deformation
   * 
   */
  void transform_shape(CvMat * src, 
                       const double scale, const double theta, 
                       const double tx, const double ty);
  /**
   * save statistical data of shape 
   * 
   */
  virtual void save(const char * datafn); // data file name - .txt file

  /**
   * load statistical data of shape
   * 
   */
  virtual void load(const char * datafn); // data file name

  /// deform shape according to learning data
  ///
  /// to be used as online shape deformation
  virtual void fit(
      // prior shape
      CvArr * _shape,
	  // grayscale image of current frame
	  const CvArr * _curr,
      // grayscale image of next frame
	  const CvArr * _next,
      // gradient image - precomputed 
      const CvArr * _grad, 
      // difference with next frame in time sequence
      const CvArr * _diff = NULL );

  /** 
   * Example shape deformation function
   * 
   * @param shape IN: input original shape 
   * @param curr  IN: input grayscale image of current frame
   * @param next  IN: input grayscale image of next frame
   */
  virtual void deform(CvMat * shape, const CvArr * curr, const CvArr * next=NULL);

};

struct CvAppearanceModel : public CvActiveShapeModel
{
  // texture information
  int nsamples;
  CvMat ** g_data;                              // data
  CvMat * g_mu;                                 // mean
  CvMat * g_covar;                              // covariance
  CvAppearanceModel():
      CvActiveShapeModel(), // initialize ASM
      nsamples(0), g_data(NULL), g_mu(NULL), g_covar(NULL)
  {}
  ~CvAppearanceModel(){
    if (g_data){
      for (int i = 0; i < nsamples; i++){
        cvReleaseMatEx(g_data[i]);
      }
    }
    cvReleaseMatEx(g_mu);
    cvReleaseMatEx(g_covar);
  }
};

//void train(const CvMat * shapelist, const IplImage ** imagelist, const int nimages);
void cvAppearanceModelLearn(
    CvAppearanceModel * aam,
    const CvMat * shape,
    const CvArr * image);

void cvAppearanceModelInference(
    CvAppearanceModel * aam,
    const CvMat * shape,
    CvArr * _curr = NULL,
    CvArr * _next = NULL,
    CvArr * _grad = NULL);



#endif //__CV_EXT_PCA_H__


