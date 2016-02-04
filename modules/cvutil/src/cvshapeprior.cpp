/** 
 * @file   cvshapeprior.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Mar 12 11:24:37 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvshapeprior.h"
#include "cvimgwarp.h"
#include <sys/types.h>
#include <sys/stat.h>

// int CvShapePrior::load(char * meanshapefn,
//                        char * meanfn, 
//                        char * pcfn)
// {
//   meanshape = rawread(meanshapefn);
//   assert(CV_MAT_TYPE(meanshape->type)==CV_32F);
//   mean = rawread(meanfn);          assert(CV_MAT_TYPE(mean->type)==CV_32F);
//   pc = rawread(pcfn);              assert(CV_MAT_TYPE(pc->type)==CV_32F);
//   // latent = rawread(latentfn);      assert(CV_MAT_TYPE(latent->type)==CV_32F);

//   assert(!proj); // NULL pointer
//   proj = cvCreateMat(1, pc->cols, CV_32F);
//   // cvPrintf(stderr, "%.1f ", mean, cvRect(0,0,10,1));
//   // cvPrintf(stderr, "%.1f ", pc, cvRect(0,0,10,10));
  
//   // CvMat * tmp = cvCreateMat(16, 16, CV_32F);
//   // meanphi = cvCreateMat(50, 50, CV_32F);

//   if (0)
//   {
//     CvMat * img = cvCreateMat(50, 50, CV_8U);
//     cvCmpS(mean, 0, img, CV_CMP_GT);
//     cvShowImageEx("Test", mean, CV_CM_GRAY); CV_WAIT();
//     cvReleaseMat(&img);
//   }
//   return 1;
// }

CvMat * CvShapePrior::shapeterm0(CvMat * phi)
{
  assert(CV_MAT_TYPE(phi->type)==CV_32F);
  int i,j,nr=phi->rows,nc=phi->cols;
  if (term0) {cvReleaseMat(&term0);}
  term0 = cvCreateMat(nr,nc, CV_32F);

  // if (1)
  // {
  //   CvMat * img = cvCreateMat(nr,nc, CV_32F);
  //   cvCalcHeaviside(phi, img, 3);
  //   cvShowImageEx("Test", img, CV_CM_GRAY); CV_WAIT();
  //   cvReleaseMat(&img);
  // }
  
  cvSet(term0, cvScalar(1));
  
  return term0;
}

CvMat * CvShapePrior::shapeterm1(CvMat * phi)
{
  if (term1) {cvReleaseMat(&term1);}
  int nr=phi->rows,nc=phi->cols;
  term1 = cvCreateMat(nr,nc,CV_32F);
  CvPoint2D32f phigc,mugc;

  // calculate gravity center of phi
  int phigcsum=0;
  {
    float * phiptr = phi->data.fl;
    int step = phi->step/sizeof(float);
    int phiyy=0,phixx=0;
    int i,j;
    for (i=0;i<nr;i++){
      for (j=0;j<nc;j++){
        if (phiptr[j]>0.) { phiyy+=i;phixx+=j;phigcsum++; }
      }
      phiptr+=step;
    }
    phigc.x=double(phixx)/double(phigcsum);
    phigc.y=double(phiyy)/double(phigcsum);
  }

  // calculate gravity center of mean shape
  int mugcsum=0;
  {
    float * muptr = meanshape->data.fl;
    int step = meanshape->step/sizeof(float);
    int muyy=0,muxx=0;
    int i,j;
    for (i=0;i<meanshape->rows;i++){
      for (j=0;j<meanshape->cols;j++){
        if (muptr[j]>0.) { muyy+=i;muxx+=j;mugcsum++; }
      }
      muptr+=step;
    }
    mugc.x=double(muxx)/double(mugcsum);
    mugc.y=double(muyy)/double(mugcsum);
  }

  // fprintf(stderr, "phigc[%d]: %f,%f\n", phigcsum, phigc.x,phigc.y);
  // fprintf(stderr, "mugc[%d]:  %f,%f\n", mugcsum,  mugc.x, mugc.y);

  float scale = sqrt(double(mugcsum)/double(phigcsum));
  float tx = (mugc.x-phigc.x*scale);//+(meanshape->cols-nc)*(scale/2.);
  float ty = (mugc.y-phigc.y*scale);//+(meanshape->rows-nr)*(scale/2.);
  // fprintf(stderr, "transform: %f,%f,%f\n", scale, tx, ty);

  CvMat * meanphi = cvCreateMat(nr,nc, CV_32F);
  CvMat * phihv = cvCreateMat(nr,nc, CV_32F);
  CvMat * phidc = cvCreateMat(nr,nc, CV_32F);
  CvMat * obj = cvCreateMat(nr,nc, CV_8U);
  CvMat * distmap = cvCreateMat(nr, nc, CV_32F);

  int nnr=meanshape->rows,nnc=meanshape->cols;
  CvMat * varphi = cvCreateMat(nnr,nnc,CV_32F);//cvCloneMat(meanshape);
  CvMat * dct_varphi = cvCreateMat(nnr,nnc,CV_32F);
  CvMat * subdct_varphi = cvCreateMat(1,256,CV_32F);
  CvMat * subdct_varphi_var = cvCreateMat(1,256,CV_32F);

  {
    float warp_p_data[9]={scale,0,tx,0,scale,ty,0,0,1};
    float invwarp_p_data[9];
    float warp_p2_data[3];
    CvMat warp_p = cvMat(3,3,CV_32F, warp_p_data);
    CvMat invwarp_p = cvMat(3,3,CV_32F, invwarp_p_data);
    CvMat warp_p2 = cvMat(3,1,CV_32F, warp_p2_data);
    cvInvert(&warp_p, &invwarp_p);
    warp_p2_data[0]=invwarp_p_data[0];
    warp_p2_data[1]=invwarp_p_data[2];
    warp_p2_data[2]=invwarp_p_data[5];
    icvWarp(phi, varphi, &warp_p2);
  }
  // cvShowImageEx("Test", varphi, CV_CM_GRAY); CV_WAIT();
  // cvPrintf(stderr, "%.2f,", varphi, cvRect(0,0,5,5));

  // static int imgiter = 0;
  static uchar varbw_data[2500];
  {
    CvMat varbw = cvMat(50,50,CV_8U,varbw_data);
    cvCmpS(varphi,0.1,&varbw,CV_CMP_GT);
    icvInitializeLevelSet(&varbw,varphi);
    //cvShowImageEx("Test", &varbw, CV_CM_GRAY);

    // FILE * fp=0;
    // char imgname[1024];
    // do {
    //   if (imgiter<10) {
    //     sprintf(imgname, "../data/tmp/000%d.png", 
    //             imgiter);
    //   }else if (imgiter<100){
    //     sprintf(imgname, "../data/tmp/00%d.png",
    //             imgiter);
    //   }else if (imgiter<1000){
    //     sprintf(imgname, "../data/tmp/0%d.png",
    //             imgiter);
    //   }else{
    //     sprintf(imgname, "../data/tmp/%d.png",
    //             imgiter);
    //   }
    //   fp = fopen(imgname,"r");
    //   imgiter++;
    // }while(fp!=0);
    // if (fp!=0){fclose(fp);}
    // cvSaveImage(imgname, &varbw);
    
    // char typestr[8];
    // char headstr[32];
    // int dim=2;
    // int arrsize[2]={0,2500};
    // FILE * fp = fopen("../data/bwset.dat", "r");
    // if (fp){ // check file existence
    //   // FORMAT:
    //   // uchar 3 10000 2500
    //   fgets(headstr,32,fp);
    //   sscanf(headstr, "%s %d %d %d",
    //          typestr,&dim,&arrsize[0],&arrsize[1]);
    //   if (arrsize[1]!=2500){
    //     fprintf(stderr, "ERROR: data file dimension error!\n");
    //     CV_WAIT(); exit(1);
    //   }
    // }
    
  }
  
  //-------------------------------------------------------
  // perform DCT upon variance of current levelset
  cvDCT(varphi, dct_varphi, CV_DXT_FORWARD);
  //cvPrintf(stderr, "%f,", varphi, cvRect(0,0,5,1));
  //cvPrintf(stderr, "%f,", dct_varphi, cvRect(0,0,5,1));
  //cvShowImageEx("Test", varphi, CV_CM_GRAY); CV_WAIT();
  {
    // copy first N rows to sub-matrix for compression
    float * dvphiptr = dct_varphi->data.fl;
    float * sdvphiptr = subdct_varphi->data.fl;
    int i,step=dct_varphi->step/sizeof(float);
    for (i=0;i<16;i++)
    {
      memcpy(sdvphiptr+i*16,dvphiptr+i*step,sizeof(float)*16);
    }
  }

  // CvMat * proj = cvCreateMat(1, pc->cols, CV_32F);
  // CvMat * X_var = cvCreateMat(proj->rows, pc->rows, CV_32F);
  assert(proj->rows==1);
  assert(pc->rows==256);
  CvMat * X_var = cvCreateMat(1, 256, CV_32F);
  {
    assert(cvGetSize(mean)==cvSize(256,1));
    cvSub(subdct_varphi, mean, subdct_varphi);
    cvMatMul(subdct_varphi, pc, proj); // dimension reduction
    //cvPrintf(stderr, "%.2f ", proj);
    cvGEMM(proj, pc, 1, mean, 1, X_var, CV_GEMM_B_T);
  }
  // cvReleaseMat(&proj);

  // int nphis_totrain = 850;
  // static int nphis = 0;
  // static float * phimatnd =
  //     (float*)malloc(16*16*sizeof(float)*nphis_totrain);

#define NUM_STAT_BUFFER 8
  static float statbuffer_data[NUM_STAT_BUFFER]={-1,};
  static int statbuffer_iter=0;
  static float X_var_vec_phi_data[256];
  CvMat statbuffer = cvMat(NUM_STAT_BUFFER,1,CV_32F,statbuffer_data);
  //-------------------------------------------------------
  // reshape feature vector to reconstruct shape
  {
    CvMat * newvarphi = cvCreateMat(nnr,nnc, CV_32F);
    CvMat * X_var_vec = cvCreateMat(16,16, CV_32F);
    CvMat X_var_vec_phi=cvMat(16,16,CV_32F,X_var_vec_phi_data);
    {
      CvMat * dct_newvarphi = cvCreateMat(nnr,nnc, CV_32F);
      cvZero(dct_newvarphi);
      // reshape
      memcpy(X_var_vec->data.fl, X_var->data.fl, sizeof(float)*256);
      // fill zeros
      {
        float * nvpptr=dct_newvarphi->data.fl;
        int i,step=dct_newvarphi->step/sizeof(float);
        for (i=0;i<16;i++){
          memcpy(nvpptr, X_var_vec->data.fl+i*16, sizeof(float)*16);
          nvpptr+=step;
        }
      }
      cvDCT(dct_newvarphi, newvarphi, CV_DXT_INVERSE);
      cvReleaseMat(&dct_newvarphi);
    }
	// cvShowImageEx("phi", newvarphi, CV_CM_GRAY); //CV_WAIT();

    cvDCT(X_var_vec, &X_var_vec_phi, CV_DXT_INVERSE);
	// cvShowImageEx("Test", &X_var_vec_phi, CV_CM_GRAY); CV_WAIT();
    
    // if ((nphis==nphis_totrain) && (phimatnd!=0))
    // {
    //   // write to file
    //   FILE * fp = fopen("../data/phi2dset.txt", "w");
    //   fprintf(fp, "%d %d %d\n", nphis_totrain,16,16);
    //   fwrite(phimatnd,sizeof(float),16*16*nphis_totrain,fp);
    //   fclose(fp);
    //   free(phimatnd);phimatnd=0;
    //   CV_WAIT();
    // }else if (nphis<nphis_totrain){
    //   // append new phi
    //   memcpy(phimatnd+(16*16*nphis),X_var_vec_phi_data,
    //          sizeof(float)*16*16);
    //   if (nphis==0){fprintf(stderr,"%f\n",phimatnd[0]);}
    //   nphis++;
    // }

    // predict shape category: open or close
    {
      CvMat newvarphi_vec = cvMat(1,16*16,CV_32F,X_var_vec_phi_data);
      float predict_result_data[2]={-1,-1};
      CvMat predict_result = cvMat(1,2,CV_32F,predict_result_data);
      // m_classifier.predict_withprior(&newvarphi_vec,&predict_result);
      m_classifier.predict_withprior(&newvarphi_vec,&predict_result);

      if (0)
      // using status buffer to stabilize output status
      {
        statbuffer_data[statbuffer_iter++%NUM_STAT_BUFFER] =
            predict_result_data[0];
        float avgval = cvAvg(&statbuffer).val[0];
        if ( (m_status!=1) && (fabs(avgval)<0.1) )
          m_status=1;
        else if ( (m_status!=0) && (fabs(avgval)>0.9) )
          m_status=0;
      }
      else{
        m_status=predict_result_data[0]<0.5;
      }
    }
    
    // warp to testing image coordinate
    {
      float warp_p_data[3]={scale,tx,ty};
      CvMat warp_p = cvMat(3,1,CV_32F, warp_p_data);
      icvWarp(newvarphi, meanphi, &warp_p);
    }
    cvReleaseMat(&newvarphi);
    cvReleaseMat(&X_var_vec);
  }
  cvReleaseMat(&X_var);

  //-------------------------------------------------------
  // enhance with a new distance transform
  {
    cvCmpS(meanphi, 0.1, obj, CV_CMP_GT);
    cvSubRS(obj, cvScalar(255), obj);
    // cvDistTransform(obj, distmap);
    cvDistTransform(obj, distmap);
    {
      float * phiptr = meanphi->data.fl;
      float * dstptr = distmap->data.fl;
      int i,j; int step = meanphi->step/sizeof(float);
      for (i=0;i<nr;i++){
        for (j=0;j<nc;j++){
          if (phiptr[j]<0.1) {phiptr[j]=-dstptr[j];}
        }
        phiptr+=step;dstptr+=step;
      }
    }
  }

  // cvShowImageEx("Test", phi, CV_CM_GRAY); CV_WAIT();
  // cvShowImageEx("Test", meanphi, CV_CM_GRAY); CV_WAIT();
  // cvCalcHeaviside(phi, phihv, 3.);
  // cvCalcDirac(phi, phidc, 3.);
  //cvShowImageEx("Test", meanphi, CV_CM_GRAY); CV_WAIT();

  // compute final shape energy term 
  {
    float * termptr = term1->data.fl;
    float * dcptr = phidc->data.fl;
    float * phiptr = phi->data.fl;
    float * muptr = meanphi->data.fl;
    int i,j;
    int step = term1->step/sizeof(float);
    assert(step=phidc->step/sizeof(float));
    assert(step=phi->step/sizeof(float));
    assert(step=meanphi->step/sizeof(float));
    for (i=0;i<nr;i++) {
      for (j=0;j<nc;j++) {
        // termptr[j]=0.1*dcptr[j]*(muptr[j]*1.1-phiptr[j]);
        // termptr[j]=dcptr[j]*(muptr[j]-phiptr[j]);
        termptr[j]=muptr[j];
      }
      termptr+=step;dcptr+=step;phiptr+=step;muptr+=step;
    }
  }
  // cvZero(term1);
  //cvShowImageEx("Test", phidc); CV_WAIT();
  //cvShowImageEx("Test", phi); CV_WAIT();
  //cvShowImageEx("Test", meanphi); CV_WAIT();
  //cvShowImageEx("Test", term1); CV_WAIT();

  cvReleaseMat(&varphi);
  cvReleaseMat(&dct_varphi);
  cvReleaseMat(&subdct_varphi);
  cvReleaseMat(&subdct_varphi_var);

  cvReleaseMat(&meanphi);
  cvReleaseMat(&phihv);
  cvReleaseMat(&phidc);
  cvReleaseMat(&obj);
  cvReleaseMat(&distmap);

  return term1;
}

//-------------------------------------------------------
// CONVERT BETWEEN SHAPE REPRESENTATIONS
//-------------------------------------------------------

// define function pointer IDs
#define CV_SHAPE_LS2DCT 0
#define CV_SHAPE_DCT2LS 1

void icvCvtShape_ls2dct(CvMat * src, CvMat * dst);
void icvCvtShape_dct2ls(CvMat * src, CvMat * dst);
void icvCvtShape(CvMat * src, CvMat * dst, int flag)
{
  typedef void (*CvCvtShapeFuncType)(CvMat *, CvMat *);
  static CvCvtShapeFuncType cvtshapefuncarr[3] = {
    &icvCvtShape_ls2dct,&icvCvtShape_dct2ls
  };
  cvtshapefuncarr[flag](src,dst);
}
void icvCvtShape_ls2dct(CvMat * src, CvMat * dst){}
void icvCvtShape_dct2ls(CvMat * src, CvMat * dst){}
