

#include "cvext/cvext.h"

#include <stdio.h>
#include <stdlib.h>

void extract_marking_data(char * fn,
                          char * filename,
                          int & firstframe,
                          const CvMat * landmarks);
void traaaaaaaaaaaaaaaaaaaaaaaaaaaaaining(CvActiveShapeModel &shapemodel);

CvRect mobiGetHeadRectFromBodyShape(CvMat * shape);

int main()
{
  const CvSize imsize = cvSize(320, 240);
  
  char filename[CV_MAXSTRLEN];
  int firstframe = 0;
  CvMat * landmarks = cvCreateMat(23, 2, CV_32S);
#if 1
  char * strMarkingDataStr = "markingData_body_frontal.txt";
  int framecounter = 210;//398;
#elif 1
  char * strMarkingDataStr = "markingData_body_frontal2.txt";
  int framecounter = 398;
#else
  char * strMarkingDataStr = "markingData_body_frontal3.txt";
  int framecounter = 450;
#endif
  extract_marking_data(strMarkingDataStr,
                       filename, firstframe, landmarks);
  
  CvCapture * capture = cvCaptureFromAVI(filename);
  if (capture==NULL){
    fprintf(stderr, "ERROR: fail to load video file!\n");
    return -1;
  }
	
  int delay = 0;
  cvNamedWindow("Test");

  //CvActiveShapeModel shapemodel;
  CvAppearanceModel shapemodel;
  traaaaaaaaaaaaaaaaaaaaaaaaaaaaaining(shapemodel);
  shapemodel.load("asm001.txt");
  //CvMat * shape = cvCreateMat(2, 23, CV_64F);
  CvMat * shape = cvCreateMat(1, 46, CV_64F);

  // get initial shape : 'shape'
  {
//CvMat * landmarks_t = 
//cvCreateMat(landmarks->cols, landmarks->rows, 
//CV_MAT_TYPE(landmarks->type));
//cvTranspose(landmarks, landmarks_t);
//cvConvert(landmarks_t, shape);
//cvReleaseMatEx(landmarks_t);

	for (int i = 0; i < landmarks->rows; i++){
      CV_MAT_ELEM(*shape, double, 0, i*2+0) = CV_MAT_ELEM(*landmarks, int, i, 0);
      CV_MAT_ELEM(*shape, double, 0, i*2+1) = CV_MAT_ELEM(*landmarks, int, i, 1);
	}

	//{
	//	IplImage * dispImage = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);cvZero(dispImage);
	//	cvDrawFrameCount(dispImage, framecounter, CV_BLUE);
	//	cvDrawLandmarks(dispImage, shape);
	//	cvShowImageEx("Test", dispImage); CV_WAIT();
	//	cvReleaseImageEx(dispImage);
	//}

  }
  
  CvAbstractTracker tracker;

  //IplImage * grayImage = cvCreateImage(imsize, IPL_DEPTH_8U, 1);
  CvMat * gray = cvCreateMat(imsize.height, imsize.width, CV_32F);
  CvMat * dx = cvCreateMat(imsize.height, imsize.width, CV_32F);
  CvMat * dy = cvCreateMat(imsize.height, imsize.width, CV_32F);
  CvMat * gradImage = cvCreateMat(imsize.height, imsize.width, CV_32F);
  
  while(1){
    cvSetCaptureProperty(capture, CV_CAP_PROP_POS_FRAMES, framecounter++);
    IplImage * rawImage = cvQueryFrame(capture);
    // cvCvtColor(rawImage, grayImage, CV_BGR2GRAY);
    if (!rawImage) {fprintf(stderr, "Info: end of video!\n"); break;}
    if (tracker.initialized()){
      tracker.update(rawImage);
      tracker.m_framecounter=framecounter;
    }else{
      tracker.initialize(rawImage);
    }

    if (tracker.m_framecounter>=firstframe) 
    {
      // get grayscale image
      //cvCvtColor(rawImage, tracker.m_currImage, CV_BGR2GRAY);
      cvConvert(tracker.m_currImage, gray);

      // compute gradient 
      cvCalcGradient<uchar>(gray, dx, dy, NULL); 
      cvCartToPolar(dx, dy, gradImage);

      {
        CV_TIMER_START();
		cvAppearanceModelLearn(&shapemodel, shape, tracker.m_currImage);
        shapemodel.fit(shape,
                       tracker.m_currImage/*CV_8U*/,
                       tracker.m_nextImage/*CV_8U*/,
                       gradImage/*CV_32F*/);
        CV_TIMER_SHOW();
      }

	  //cvDrawLandmarks(tracker.m_currImage, shape);
	  //cvShowImageEx("Test", tracker.m_currImage); CV_WAIT();
	
	  //if (0)
	  {
        IplImage * dispImage = cvCloneImage(rawImage);
        cvDrawFrameCount(dispImage, framecounter, CV_BLUE);
        cvDrawLandmarks(dispImage, shape, CV_RED);
        CvRect faceRect = mobiGetHeadRectFromBodyShape(shape);
        cvRectangle(dispImage, cvPoint(faceRect.x, faceRect.y),
                    cvPoint(faceRect.x+faceRect.width, faceRect.y+faceRect.height),
                    CV_GREEN, 1);
        cvShowImageEx("Test", dispImage); CV_WAIT2(10);
        cvReleaseImageEx(dispImage);
	  }
    } // end of if 
	else{
      IplImage * dispImage = cvCloneImage(rawImage);
      cvDrawFrameCount(dispImage, framecounter, CV_BLUE);
      cvShowImageEx("Test", dispImage); CV_WAIT2(10);
      cvReleaseImageEx(dispImage);
	}

    int key = cvWaitKey(delay)&0xff;
    if (key==27){
      break;
    }else if (key==' '){
      if (delay){ delay = 0; }else{ delay = 30; }
    }else if (key=='f'){ // skip to next frame
    }else if (key!=0xff){
      fprintf(stderr, "Warning: Unknown key press : %c\n", key);
    } // end of key press processing
  } // end of while loop

  cvDestroyWindow("Test");
  //cvReleaseImageEx(grayImage);
  cvReleaseMatEx(gray);
  cvReleaseMatEx(dx);
  cvReleaseMatEx(dy);
  cvReleaseMatEx(gradImage);

  cvReleaseMatEx(landmarks);
  cvReleaseMatEx(shape);
  //cvReleaseImageEx(rawImage);
  return 0;
}

void extract_marking_data(char * fn,
                          char * filename,
                          int & firstframe,
                          const CvMat * landmarks)
{
  FILE * fp = fopen(fn, "r"); 
  if (!fp){
    fprintf(stderr, "Error: fail to open file %s\n", fn);
    exit(1);
  }
  fscanf(fp, "%s", filename);
  fscanf(fp, "%d", &firstframe);
  for (int i = 0; i < 23; i++){
    fscanf(fp, "%d %d",
           &(CV_MAT_ELEM(*landmarks, int, i, 0)),
           &(CV_MAT_ELEM(*landmarks, int, i, 1)));
	CV_MAT_ELEM(*landmarks, int, i, 1) = 240-CV_MAT_ELEM(*landmarks, int, i, 1);
  }
  fclose(fp);
}

void traaaaaaaaaaaaaaaaaaaaaaaaaaaaaining(CvActiveShapeModel &shapemodel)
{
	FILE * fp = fopen("markingData_body_frontal_flipped.txt", "r");
	char fn[1024]; int no; 
	int x, y;

	const int N = 12;
	const int M = 23*2;

	CvMat * data = cvCreateMat(N, M, CV_64F);

	// get data matrix for training
	for (int ff = 0; ff < N; ff++){
		fscanf(fp, "%s %d\n", fn, &no);
		for (int i = 0; i < 23; i++){
			fscanf(fp, "%d %d\n", &x, &y);
			cvmSet(data, ff, i*2+0, x);
			cvmSet(data, ff, i*2+1, y);
		}fscanf(fp, "\n");
	}

	shapemodel.train(data);
	shapemodel.save("asm001.txt");

	cvReleaseMatEx(data);
	fclose(fp);
}

/**
 * Structure of head section contour:
 *
 *        [11]
 *       /    \
 *     [09]   [13]
 *      |      |
 *     [07]   [15]
 *
 * We extract rectangle at point [09] and [15]
 * 
 * @param shape in: full frontal body shape
 * 
 * @return out: head rectangle
 */
CvRect mobiGetHeadRectFromBodyShape(CvMat * shape)
{
  assert(shape->rows==1);
  assert(shape->cols/2==23);
  const int npoints = shape->cols/2;
  return cvRect(cvRound(shape->data.db[20]), // x at [09]
                cvRound(shape->data.db[21]), // y at [09]
                cvRound(shape->data.db[30]-shape->data.db[20]), // width between [19]&[15]
                cvRound(shape->data.db[31]-shape->data.db[21]));// height between [19]&[15]
}

