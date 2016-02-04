

#include "cvext/cvext.h"

#include <stdio.h>
#include <stdlib.h>

#include "cvext_pca.h"

int main()
{
	FILE * fp = fopen("markingData_body_frontal_flipped.txt", "r");
	char fn[1024]; int no; 
	int x, y;

	const int N = 12;
	const int M = 23*2;

	CvMat * data = cvCreateMat(N, M, CV_64F);
	CvMat * mean = cvCreateMat(1, M, CV_64F);

	// get data matrix for training
	for (int ff = 0; ff < N; ff++){
		fscanf(fp, "%s %d\n", fn, &no);
		for (int i = 0; i < 23; i++){
			fscanf(fp, "%d %d\n", &x, &y);
			cvmSet(data, ff, i*2+0, x);
			cvmSet(data, ff, i*2+1, y);
		}fscanf(fp, "\n");
	}

	const int ncols = 320, nrows = 240;
	IplImage * dispImage = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3); 
	cvSet(dispImage,cvScalarAll(255));
	
	CvPrincipleComponentAnalysis pca;
	CV_TIMER_START();
	pca.set_data(data);
	pca.back_project();
	CV_TIMER_SHOW();
	cvCopy(pca.mean, mean);

	CvMat subdatahdr;
	CvMat * subdata = cvGetSubRect(data, &subdatahdr, cvRect(0,0,M,1));
	CvMat subdata2hdr, meanhdr;
	CvMat * subdata2 = cvReshape(subdata, &subdata2hdr, 2);
	CvMat * mean2 = cvReshape(mean, &meanhdr, 2);
	double scale, theta, tx,ty;
	cvShapeFitting(mean2, subdata2, scale, theta, tx, ty);

	CvMat * subdata3 = cvCreateMat(2, 23, CV_64F);
	for (int i = 0; i < 23; i++){
		cvmSet(subdata3, 0, i, cvmGet(subdata, 0, i*2+0));
		cvmSet(subdata3, 1, i, cvmGet(subdata, 0, i*2+1));
	}
	cvShapeTransform2(subdata3,scale, theta, tx, ty);
	for (int i = 0; i < 23; i++){
		cvmSet(subdata, 0, i*2+0, cvmGet(subdata3, 0, i));
		cvmSet(subdata, 0, i*2+1, cvmGet(subdata3, 1, i));
	}
	cvReleaseMatEx(subdata3);

	// compare landmarks
	if (0)
	for (int i = 0; i < 12; i++){
		for (int j = 0; j < 23; j++){
			cvCircle(dispImage, cvPoint(cvmGet(data, i, j*2), cvmGet(data, i, j*2+1)), 2, cvScalar(255,0,0), -1);
			//cvCircle(dispImage, cvPoint(cvmGet(orig, i, j*2), cvmGet(data, i, j*2+1)), 2, cvScalar(0,0,255), -1);
		}
	}

	for (int j = 0; j < 23; j++){
		cvCircle(
			dispImage, cvPoint(cvmGet(mean, 0, j*2), cvmGet(mean, 0, j*2+1)), 
			1, cvScalar(255,0,0), -1);
	}
	for (int j = 0; j < 23; j++){
		cvCircle(
			dispImage, cvPoint(cvmGet(subdata, 0, j*2), cvmGet(subdata, 0, j*2+1)), 
			1, cvScalar(0,0,255), -1);
	}

	cvNamedWindow("Test");
	cvShowImageEx("Test", dispImage); cvWaitKey(0);

	fclose(fp);
	cvReleaseImageEx(dispImage);

	cvReleaseMatEx(data);
	cvReleaseMatEx(mean);
	return 0;
}

