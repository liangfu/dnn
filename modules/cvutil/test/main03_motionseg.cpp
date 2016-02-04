/**
 * @file   main03_motionseg.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Thu Nov 29 09:55:01 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvext.h"

//CV_INLINE
//void cvGetRegionSum(CvArr * _integral, CvRect roi, int & sum)
//{
//	CvMat header; CvMat * integral = cvGetMat(_integral, &header);
//	assert(CV_MAT_TYPE(integral->type)==CV_32S);
//	int lt = CV_MAT_ELEM(*integral, int, roi.y, roi.x);
//	int rt = CV_MAT_ELEM(*integral, int, roi.y, roi.x+roi.width);
//	int rb = CV_MAT_ELEM(*integral, int, roi.y+roi.height, roi.x+roi.width);
//	int lb = CV_MAT_ELEM(*integral, int, roi.y+roi.height, roi.x);
//	sum = rb-(rt+lb-lt); 
//}
//
//CV_INLINE
//void cvGetRegionSum(CvArr * _integral, CvRect roi, double & sum)
//{
//	CvMat header; CvMat * integral = cvGetMat(_integral, &header);
//	assert(CV_MAT_TYPE(integral->type)==CV_64F);
//	double lt = CV_MAT_ELEM(*integral, double, roi.y, roi.x);
//	double rt = CV_MAT_ELEM(*integral, double, roi.y, roi.x+roi.width);
//	double rb = CV_MAT_ELEM(*integral, double, roi.y+roi.height, roi.x+roi.width);
//	double lb = CV_MAT_ELEM(*integral, double, roi.y+roi.height, roi.x);
//	sum = rb-(rt+lb-lt); 
//}

//--------------------------------
// imgSrc : source image
// roi : region of interest
// pt : seed point
// qSeq : queue, as list of sequence points
// return value: area of the propagated region
inline int cvSweep2(CvArr * src, CvRect roi, CvPoint pt, 
				   CvPoint pts[] ) // queue
{
  IplImage header; CvMat regionhdr;
  IplImage * imgSrc = cvGetImage(src, &header);

  static const int nDx[4] = {0,1,0,-1};
  static const int nDy[4] = {-1,0,1,0};
  int m_imgSize = imgSrc->width*imgSrc->height;
  int xMin, xMax, yMin, yMax;
  xMin = roi.x; xMax = roi.x+roi.width;
  yMin = roi.y; yMax = roi.y+roi.height;
  int nCurrX=xMin, nCurrY=yMin;
  int idxCurr;
  int nStart=0, nEnd=1; // position in Queue
#define xxx 0
#if xxx
  cvClearSeq(qSeq);
  cvSeqPush(qSeq, &cvPoint(pt.x, pt.y));
#else
  int nSeeds=0;
  memset(pts,0,19200);pts[0]=cvPoint(pt.x, pt.y);
#endif
  // position of the 4 connected point
  int xx = 0;
  int yy = 0;
  int k = 0;

  int offset[4] = {
    imgSrc->widthStep*nDy[0] + nDx[0], // U
    imgSrc->widthStep*nDy[1] + nDx[1], // R
    imgSrc->widthStep*nDy[2] + nDx[2], // D
    imgSrc->widthStep*nDy[3] + nDx[3]  // L
  };

  uchar * pDataSrc = (unsigned char*)imgSrc->imageData;
  while (nStart<nEnd && nEnd<m_imgSize)
  {
#if xxx
    // position of the current seed
    nCurrX = CV_GET_SEQ_ELEM(CvPoint, qSeq, nStart)->x;
	nCurrY = CV_GET_SEQ_ELEM(CvPoint, qSeq, nStart)->y;
#else
    nCurrX = pts[nStart].x;nCurrY = pts[nStart].y;
#endif
	idxCurr = -1; 
	//if (1) {fprintf(stderr, "%d %d; ", nCurrX, nCurrY);}

    // Search the 4 connected point of current seed
    for (k=0; k<4; k++) 
    {    
      xx = nCurrX + nDx[k];
      yy = nCurrY + nDy[k];
      if (xx<xMin || xx>xMax || yy<yMin || yy>yMax)
      {
        continue; // not in range of ROI
      }
      if (idxCurr<0)
      {
        idxCurr = imgSrc->widthStep * nCurrY + nCurrX;
      }

	  
      //if ( pDataSrc[ idxCurr + offset[k] ] == 1 ) 
	  if ( pDataSrc[ idxCurr + offset[k] ] != 0 ) 
	  //if ( pDataSrc[ idxCurr + offset[k] ] <= 3 ) 
      {
        // pixel in (xx,yy) to stack
#if xxx
	    cvSeqPush(qSeq, &cvPoint(xx, yy));
#else
	    pts[nSeeds++]=cvPoint(xx,yy);
#endif 
#undef xxx
		pDataSrc[ idxCurr + offset[k] ] = 0;

        ++nEnd;  // Stack end point move forward
        if (nEnd>=m_imgSize)
        {
          break;
        }
	  }
    }
    ++nStart;
  }//fprintf(stderr, "\n");
  return nSeeds;
}

//void cvSeededBinaryDilate(CvArr * _img, CvPoint seed)
//{
//	IplImage imghdr;
//IplImage * imgSrc = cvGetImage(_img,&imghdr);
//
//  static const int nDx[4] = {0,1,0,-1};
//  static const int nDy[4] = {-1,0,1,0};
//  int m_imgSize = imgSrc->width*imgSrc->height;
//  int xMin, xMax, yMin, yMax;
//  xMin = 0; xMax = imgSrc->width;
//  yMin = 0; yMax = imgSrc->height;
//  int nCurrX=xMin, nCurrY=yMin;
//  int idxCurr;
//  int nStart=0, nEnd=1; // position in Queue
//  static CvPoint pts[19200];pts[0]=seed;
//
//  // position of the 4 connected point
//  int xx = 0;
//  int yy = 0;
//  int k = 0;
//
//  int offset[4] = {
//    imgSrc->widthStep*nDy[0] + nDx[0], // U
//    imgSrc->widthStep*nDy[1] + nDx[1], // R
//    imgSrc->widthStep*nDy[2] + nDx[2], // D
//    imgSrc->widthStep*nDy[3] + nDx[3]  // L
//  };
//int nSeeds=0;
//  uchar * pDataSrc = (unsigned char*)imgSrc->imageData;
//  while (nStart<nEnd && nEnd<m_imgSize)
//  {
//	  //cvShowImage("Test", imgSrc); CV_WAIT();
//    
//	  nCurrX = pts[nStart].x;
//	nCurrY = pts[nStart].y;
//	idxCurr = -1; 
//
//    // Search the 4 connected point of current seed
//    for (k=0; k<4; k++) 
//    {    
//      xx = nCurrX + nDx[k];
//      yy = nCurrY + nDy[k];
//      if (xx<xMin || xx>xMax || yy<yMin || yy>yMax)
//      {
//        continue; // not in range of ROI
//      }
//      if (idxCurr<0)
//      {
//		  idxCurr = imgSrc->widthStep * nCurrY + nCurrX;//if (pDataSrc[idxCurr]==0){return;}
//      }
//	  
//      if (pDataSrc[ idxCurr + offset[k] ] == 0 && (pDataSrc[idxCurr]>0&&pDataSrc[idxCurr]<=15)){
//		//pDataSrc[ idxCurr + offset[k] ] = 0;
//	    pDataSrc[idxCurr+ offset[k]] = pDataSrc[idxCurr]+1;
//	    pts[nSeeds++]=cvPoint(xx,yy);
//
//        ++nEnd;  // Stack end point move forward
//        if (nEnd>=m_imgSize)
//        {
//          break;
//        }
//	  }
//    }
//    ++nStart;
//  }
//
//}
//

//----------------------------------------------
// imgSrc: binary image all nonzero values set to 1
// integral : integral image of the binary image
// roi : region of interest for finding motion
// pts : pointer to list of seed points
// bsize : block size as point searching critiria
// return value: number of seed points
int genseed(
    IplImage * imgSrc,
    CvMat * integral,
    const CvRect roi,
    const int bsize = 3
            )
{
  const int maxpoints = 256;
  static CvPoint seedset[256]; memset(seedset, 0,sizeof(CvPoint)*maxpoints);
  int nSeeds = 0;

  static uchar pDistData[19200];
  CvMat pDist = cvMat(120,160,CV_8U,pDistData); 
  memset(pDist.data.ptr,0,19200);

  static int tS[1024]={0,};
  static CvPoint pts[19200];

  //memcpy(pDist.data.ptr, imgSrc->imageData, 19200);
  for (int i = roi.y; (i<roi.height-bsize)&&(nSeeds!=maxpoints); i+=bsize){
    for (int j = roi.x; j<roi.width-bsize; j+=bsize){
	  int block_area=0;
	  cvGetRegionSum(integral,cvRect(j,i,bsize,bsize),block_area);
      if (block_area!=0) {
        CvPoint seed=cvPoint(j+bsize/2,i+bsize/2);
		if (CV_IMAGE_ELEM(imgSrc, uchar, seed.y, seed.x)==0){continue;}// next block if zero

        // sweep collected region
        int total =cvSweep2(&pDist, roi, seed, pts); 

		//if (total>bsize)
		{
          seedset[nSeeds++]=seed;
		}
		if (nSeeds==maxpoints) {break;}
      }
    }
  }
//return nSeeds;

cvSet(&pDist, cvScalar(255), &pDist);
//cvShowImage("Test", &pDist); CV_WAIT();//cvWaitKey(60)==27?exit(1):0;
  memcpy(pDist.data.ptr, imgSrc->imageData, 19200);
  for (int i = 0; i < nSeeds; i++){
    cvSeededBinaryDilate(&pDist, seedset[i], 7);
  }
cvSet(&pDist, cvScalar(255), &pDist);
//cvShowImage("Test", &pDist); CV_WAIT();//cvWaitKey(60)==27?exit(1):0;

  for (int i = 0; i < nSeeds; i++){
    int total = cvSweep2(&pDist, roi, seedset[i], pts);
	if (total>0){
		cvAnd(imgSrc, &pDist, imgSrc);
		cvSet(imgSrc, cvScalar(255), imgSrc);
		//cvShowImage("Test", imgSrc); CV_WAIT();
		cvSet(imgSrc, cvScalar(1), imgSrc);
	}
  }

  return nSeeds;
}

int main()
{
  IplImage * motionImage;
  cvNamedWindow("Test");
char fname[CV_MAXSTRLEN];
  //for (int i = 1; i <= 5; i++)
int i=5;
  {
      sprintf(fname, "../data/motion%d.bmp",i);
	  motionImage = cvLoadImage(fname, 0); // grayscale
	  if (!motionImage){
		  fprintf(stderr, "Error: %s: %d: image file not found", 
			  __FILE__, __LINE__);
	  }
	  assert(cvGetElemType(motionImage)==CV_8U);

	  // original data
	  //cvSet(motionImage, cvScalar(255), motionImage);
	  //cvShowImage("Test", motionImage); CV_WAIT();
CV_TIMER_START();
	  static int integral_data[19481]={0,}; 
	  memset(integral_data,0,sizeof(int)*19481);
	  CvMat integral = 
		  cvMat(motionImage->height+1, motionImage->width+1, CV_32S, integral_data);
	  cvIntegral(motionImage, &integral);
	  CvRect roi = cvRect(2,10,120,117-10);
	  int nSeeds = genseed(motionImage, &integral, roi);
CV_TIMER_SHOW();

	  cvSet(motionImage, cvScalar(64), motionImage);
	  cvRectangle(motionImage, cvPoint(roi.x,roi.y), 
		  cvPoint(roi.x+roi.width,roi.y+roi.height), cvScalar(128));
	  cvShowImage("Test", motionImage); cvWaitKey();
  }
  cvDestroyWindow("Test");
  return 0;
}
