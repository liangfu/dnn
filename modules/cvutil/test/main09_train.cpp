
#include "cvext.h"
#include "opencvx/cvskincolorgauss.h"
#include "opencvx/cvskincolorgmm.h"

const char * trainlist[] = {"palm_jc.png","palm_jiang.png","palm_jason.png"};
const char * traindir = "../data/";
const int nsamples = 3;

void cbMouse(int evt, int x, int y, int flags, void* param);

int main()
{
	char trainfn[1024];
	//IplImage * img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3); 
	//cvZero(img);
	cvNamedWindow("Test",0);
	for (int i = 0; i < nsamples; i++)
	{
		strcpy(trainfn, traindir);
		strcat(trainfn, trainlist[i]);

		IplImage * img = cvLoadImage(trainfn, 1);
		IplImage * mask = cvCreateImage(cvGetSize(img), 8, 1);
		IplImage * probs = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1);
CV_TIMER_START();
		cvSkinColorGmm(img, mask,0.8,probs); 
CV_TIMER_SHOW();
		cvSet(mask, cvScalar(255),mask);
		//cvSetMouseCallback("Test",cbMouse, img);
		//while (1) {cvShowImage("Test", mask); CV_WAIT2(30);}
		cvShowImage("Test", img); CV_WAIT();
		cvShowImage("Test", mask); CV_WAIT();
		cvShowImageEx("Test", probs); CV_WAIT();
	}
	cvDestroyWindow("Test");
	return 0;
}

void cbMouse(int evt, int x, int y, int flags, void* param)
{
	if (CV_EVENT_LBUTTONDOWN&evt){  // point
	//if (CV_EVENT_FLAG_LBUTTON&flags){ // drag
		fprintf(stderr, "(%d,%d)\n", x, y);
		cvCircle((IplImage*)param, cvPoint(x,y),1,CV_BLACK,-1);

	}
}