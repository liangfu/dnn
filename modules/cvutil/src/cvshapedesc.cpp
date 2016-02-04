/**
 * @file   cvshapedesc.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Apr  1 11:34:15 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvshapedesc.h"

CVAPI(void) cvExtractClosedContour(
    CvMat * bw, CvSeq * contour, CvMemStorage * storage)
{
  // CV_SHOW(bw);
  assert(CV_MAT_TYPE(bw->type)==CV_8U);
  int i,j,nr=bw->rows,nc=bw->cols;
  CvMat * vi = cvCreateMat(nr,nc,CV_8U); // visited list
  cvZero(vi);
  int bwstep=bw->step/sizeof(uchar);
  int vistep=vi->step/sizeof(uchar);
  uchar * bwptr=bw->data.ptr+(nr-1)*bwstep;
  uchar * viptr=vi->data.ptr+(nr-1)*vistep;
  // ---------------------------------
  // find initial seed from bottom-left corner
  // ---------------------------------
  int seed=-1;
  for (i=nr-1;i>-1;i--){
    // for (j=0;j<MIN(nc,nr-i);j++){
    for (j=0;j<nc;j++){
      if (bwptr[j]) {viptr[j]=1;seed=i*nc+j;break;}
    }
    bwptr-=bwstep; viptr-=vistep;
    if (seed>0){break;}
  }
  // ---------------------------------
  // propagate with the seed
  // ---------------------------------
  // int offset0[4]={nc,-1,-nc,1};
  // int offset1[4]={-1,-nc,1,nc};
  // int offset0[6]={nc,nc-1,-1,-nc,-nc+1,1};
  // int offset1[6]={nc-1,-1,-nc,-nc+1,1,nc};
  int offset0[8]={nc,nc-1,-1,-nc-1,-nc,-nc+1,1,nc+1};
  int offset1[8]={nc-1,-1,-nc-1,-nc,-nc+1,1,nc+1,nc};
  int * iq = new int[nr*nc];
  int iqf=0,iqb=0,curr,nbr0,nbr1,nbr0y,nbr0x,nbr1y,nbr1x;
  iq[iqb++]=seed; // push initial seed
  bwptr=bw->data.ptr;
  viptr=vi->data.ptr;
  while (iqf!=iqb)
  {
    curr=iq[iqf++];//viptr[curr]=255;
    for (i=0;i<8;i++)
    {
      nbr0=curr+offset0[i];nbr0y=nbr0/nc;nbr0x=nbr0%nc;
      nbr1=curr+offset1[i];nbr1y=nbr1/nc;nbr1x=nbr1%nc;
      if (viptr[nbr0]||viptr[nbr1]||
          (nbr0y>(nr-1))||(nbr0x>(nc-1))||
          (nbr1y>(nr-1))||(nbr1x>(nc-1))||
          (nbr0y<1)||(nbr0x<1)||
          (nbr1y<1)||(nbr1x<1)) {continue;} // not visited before!
      // if (viptr[nbr0] || // not visited before!
      //     (((bwptr+nbr0y*bwstep)[nbr0x]==
      //       (bwptr+(nbr0y+1)*bwstep)[nbr0x])
      //      &&
      //      ((bwptr+nbr0y*bwstep)[nbr0x]==
      //       (bwptr+nbr0y*bwstep)[(nbr0x+1)])))
      // {
      //   continue;
      // } 
      // if (((bwptr+nbr0y*bwstep)[nbr0x]>0) xor
      //     ((bwptr+nbr1y*bwstep)[nbr1x]>0))
      if (((bwptr+nbr0y*bwstep)[nbr0x]>0) ^
          ((bwptr+nbr1y*bwstep)[nbr1x]>0))
      {
        // fprintf(stderr, "(%d,%d):%d,(%d,%d):%d\n",
        //         nbr0y,nbr0x,
        //         (bwptr+nbr0y*bwstep)[nbr0x],
        //         nbr1y,nbr1x,
        //         (bwptr+nbr1y*bwstep)[nbr1x]);

        iq[iqb++]=((bwptr+nbr0y*bwstep)[nbr0x]!=0)?nbr0:nbr1;
        viptr[iq[iqb-1]]=255;

        // break; // assign propagate direction
        if (iqb<15){break;}
      }
    }
    if ((iqb+9)>(nr*nc)) {break;}
    // fprintf(stderr, "--\n");
    // if((iqf%5)==0){CV_SHOW(vi);}
  }
  for (i=0;i<iqb;i++){
    // cvSeqPush(contour,&iq[i]);
    CvPoint2D32f pt=cvPoint2D32f(iq[i]%nc,iq[i]/nc);
    cvSeqPush(contour,&pt);
    // fprintf(stderr, "(%d,%d),",iq[i]/nc,iq[i]%nc);
  }
  // CV_SHOW(vi);
  delete [] iq;
  cvReleaseMat(&vi);
}

CVAPI(void) cvExtractFourierDescriptor(
    CvMat * contour, CvMat * fdesc)
{
  int nr=contour->rows,fdsize=fdesc->rows;
  CvMat * contourfft = cvCreateMat(nr,1,CV_32FC2);
  cvDFT(contour, contourfft, CV_DXT_FORWARD);
  {
    CvPoint2D32f p = CV_MAT_ELEM(*contourfft,CvPoint2D32f,1,0);
    float scalecoeff=sqrt(p.x*p.x+p.y*p.y);
    cvScale(contourfft, contourfft,1./scalecoeff);
  }
  memcpy(fdesc->data.ptr,
         contourfft->data.ptr+sizeof(float)*2,
         sizeof(float)*(fdsize/2)*2);
  memcpy(fdesc->data.ptr+sizeof(float)*fdsize,
         contourfft->data.ptr+(sizeof(float)*(nr-fdsize/2)*2),
         sizeof(float)*(fdsize/2)*2);
  cvReleaseMat(&contourfft);
}

int CvFourierDescriptor::train(CvMat ** imgdata, CvMat * _response)
{
  const int N = _response->rows;
  int i;
  CvMemStorage ** storages = new CvMemStorage * [N];
  CvSeq ** contours = new CvSeq * [N];
  CvMat * train_data = cvCreateMat(N,fdsize*2,CV_32F);
  
  for (i=0;i<N;i++)
  {
    storages[i] = cvCreateMemStorage(0);
    contours[i] =
        cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storages[i]);
    cvExtractClosedContour(imgdata[i], contours[i], storages[i]);

    CvMat * contour = cvCreateMat(contours[i]->total, 1, CV_32FC2);
    cvCvtSeqToArray(contours[i], contour->data.i);
    CvMat * fdesc = cvCreateMat(fdsize, 1, CV_32FC2);
    cvExtractFourierDescriptor(contour, fdesc);
    memcpy(train_data->data.fl+i*fdsize*2,fdesc->data.fl,
           sizeof(float)*fdsize*2);

    cvReleaseMemStorage(&storages[i]);
    cvReleaseMat(&contour);
    cvReleaseMat(&fdesc);
  }

  //cvPrintf(stderr, "%.2f,", train_data);
  {
    assert(CV_MAT_TYPE(_response->type)==CV_32S);
    CvMat * response = cvCreateMat(N, 1, CV_32S);
    memcpy(response->data.i,_response->data.i,sizeof(int)*N);
    
    m_classifier.train(train_data, response);

    cvReleaseMat(&response);
  }

  {
    // int K=2;
    // CvMat * test = cvCreateMat(1,fdsize*2,CV_32F);
    // CvMat * result = cvCreateMat(1, K, CV_32F);
    // for (i=0;i<N;i++)
    // {
    //   memcpy(test->data.fl,
    //          train_data->data.fl+fdsize*2*i,
    //          sizeof(float)*fdsize*2);
    //   m_classifier.predict(test,result);
    //   cvPrintf(stderr, "%f,", result);
    // }
    // cvReleaseMat(&test);
    // cvReleaseMat(&result);

    // CvMat * result = cvCreateMat(N, 2, CV_32F);
    // m_classifier.predict(train_data, result);
    // cvPrintf(stderr, "%.2f,", result);
    // cvReleaseMat(&result);
  }

  delete [] contours;
  delete [] storages;
  cvReleaseMat(&train_data);
  return 1;
}

int CvFourierDescriptor::predict(CvMat * sample, CvMat * result)
{
  CvMemStorage * storage = cvCreateMemStorage(0);
  CvSeq * contour =
      cvCreateSeq(CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage);
  cvExtractClosedContour(sample, contour, storage);

  if (contour->total>22)
  {
    CvMat * contourmat = cvCreateMat(contour->total, 1, CV_32FC2);
    cvCvtSeqToArray(contour, contourmat->data.i);
    CvMat * fdesc = cvCreateMat(fdsize, 1, CV_32FC2);
    cvExtractFourierDescriptor(contourmat, fdesc);

    CvMat * featvec = cvCreateMat(1, fdsize*2, CV_32F);
    // transpose
    memcpy(featvec->data.fl,fdesc->data.fl,sizeof(float)*fdsize*2); 
    m_classifier.predict(featvec, result);

    cvReleaseMat(&contourmat);
    cvReleaseMat(&fdesc);
    cvReleaseMat(&featvec);
  }else{
    cvReleaseMemStorage(&storage);
    return 0;
  }

  cvReleaseMemStorage(&storage);
  return 1;
}

