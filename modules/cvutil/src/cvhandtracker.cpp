/**
 * @file   cvext_hand.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Dec 10 14:16:16 2012
 * 
 * @brief  
 * 
 * 
 */

#include "cvhandtracker.h"

void CvHandTracker::update()
{
// CV_TIMER_START();
  CvPWPTracker::update();
// CV_TIMER_SHOW();

  if (CvPWPTracker::initialized()) {
    assert(phi);
    if (!registration()) { assert(false); }
    if (!segmentation()) { assert(false); }
  }else{
    CvPoint2D32f center=cvPoint2D32f(0,0);
    float size=0, score=0;
    // if (m_detector.detect(m_t.m_currImage,center,size,score))
    if (m_detector.cascade_detect(m_t.m_currImage,center,size))
    {
      CvBox2D box;
      box.center=center;
      box.angle=0;
      box.size=cvSize2D32f(size*0.6,size*0.8);
      CvRect roi=cvBox2DToRect(box);
      CvPWPTracker::initialize(roi);
    }
  }

  // if (!CvParticleFilter::initialized()){
  // if (!CvParticleFilter::initialized() && initialized()){
  //   CvRect roi = cvBox2DToRect(m_outerbox);
  //   CvParticleFilter::initialize(roi);
  // }else
  if (CvParticleFilter::initialized()){
    CvParticleFilter::observe();
  }else{
    CvPoint2D32f center=cvPoint2D32f(0,0);
    float size=0, score=0;

    // CvParticleFilter::initialize(cvRect(103-10,37-10,24+20,30+20));
    if (m_detector.cascade_detect(m_t.m_currImage,center,size))
    {
      CvBox2D box;
      box.center=center;
      box.angle=0;
      // box.size=cvSize2D32f(size*0.6,size*0.8);
      box.size=cvSize2D32f(size*1.2,size*1.4);
      CvRect roi=cvBox2DToRect(box);
      CvParticleFilter::initialize(roi);
    }
  }

}

#if defined(WITH_TZG)

#define USE_CASCADE_DETECTOR

int CvHandTracker::myTrack(int has_fg)
{
	if (!CvPWPTracker::initialized() && valid())
	{
		// if (has_fg)
		{
#ifdef USE_CASCADE_DETECTOR
			CvPoint2D32f center=cvPoint2D32f(0,0);
			float size=0, score=0;
			CvRect roi;
			if (m_detector.cascade_detect(m_t.m_currImage,center,size))
			{
				// fprintf(stderr,
				//         "detected: at[%.1f,%.1f] "
				//         "size: %.1f, scoring: %.1f\n",
				//         center.x, center.y, size, score);
				CvBox2D box;
				// center.y+=4.*(size/25.);
				box.center=center;
				box.angle=0;
				// box.size=cvSize2D32f(size*0.8+16,size*1.2+16);
				box.size=cvSize2D32f(size*0.8,size*1.2);
				roi=cvBox2DToRect(box);
			}
#else
			CvBox2D box = m_validator.get_roi();
			CvRect roi = cvBox2DToRect(box);
#endif

			CvPWPTracker::myInitialize(roi);
        }
	}

	// if (CvParticleFilter::initialized()) {
	if (CvPWPTracker::initialized()) {
		assert(phi);
//CV_TIMER_START();
		if (!myRegistration()) { assert(false); }
//CV_TIMER_SHOW();
//CV_TIMER_RESTART();
		if (!mySegmentation()) { assert(false); }
//CV_TIMER_SHOW();
		// CvParticleFilter::
        //   observe(m_t.m_currColorImage, has_fg?maskImage:0);
	}

	//CvPWPTracker::myDisplay();
	checkTrackFalse();
	if(!CvPWPTracker::initialized())
	{
		m_validator.reset();		
	}

	return 1;
}


void CvHandTracker::myUpdate()
{
#ifdef USE_CASCADE_DETECTOR
	CvPWPTracker::update();
	if (CvPWPTracker::initialized()) {
		myTrack(0);
		return;
	}
	else
	{
		CvPoint2D32f center=cvPoint2D32f(0,0);
		float size=0, score=0;
		if (m_detector.cascade_detect(m_t.m_currImage,center,size))
		{
			// fprintf(stderr,
			//         "detected: at[%.1f,%.1f] "
			//         "size: %.1f, scoring: %.1f\n",
			//         center.x, center.y, size, score);
			CvBox2D box;
			// center.y+=4.*(size/25.);
			box.center=center;
			box.angle=0;
			// box.size=cvSize2D32f(size*0.8+16,size*1.2+16);
			box.size=cvSize2D32f(size*0.8,size*1.2);
			CvRect roi=cvBox2DToRect(box);
			CvPWPTracker::myInitialize(roi);
		}
		return;
	}

#else
	//CvPWPTracker::update();
	if (CvPWPTracker::initialized()) { myTrack(0); return; }

	CvSize imsize = CvPWPTracker::m_imsize;
	CvMemStorage * storage = cvCreateMemStorage(0);
	IplImage * imgSegMask =
		cvCreateImage(imsize, IPL_DEPTH_32F, 1);
	CvSeq * seq =
		cvSegmentMotion(m_t.m_mhiImage, imgSegMask, storage, 256, 16);
	CvRect roi;
	int maxloc=0; double maxval = -1.; double tmaxval=-1.;
	int err;

	CvMat * mask = cvCreateMat(imsize.height, imsize.width, CV_8U);

	for (int i = 0; i < seq->total; i++)
	{
		roi = ((CvConnectedComp*)cvGetSeqElem(seq, i))->rect;
		if ( (roi.height+roi.width) > (m_iLinMultiplier<<3) )//if ( (roi.height+roi.width) > (m_iLinMultiplier<<5) ) // 25x25
		{
			cvCmpS(imgSegMask, i+1, mask, CV_CMP_EQ);
			cvSet(mask, cvScalar(1), mask);
			tmaxval = cvSum(mask).val[0];
			// more than `thres` points
			if ( (tmaxval>maxval)&&(tmaxval>(10*m_iSqrMultiplier)) ) //if ( (tmaxval>maxval)&&(tmaxval>(100*m_iSqrMultiplier)) ) 
			{
				maxval = tmaxval;
				maxloc = i;
			}
		}
	}

	// -------------------------------------------------------
	// with motion detected
	if (maxval>0)                                 
	{
		roi = ((CvConnectedComp*)cvGetSeqElem(seq, maxloc))->rect;
		const int newh = roi.width/3.*4.;
		if (newh<=roi.height) {
			roi.height = newh;
		}else{
			if (newh+roi.y<imsize.height){
				roi.y = MAX(roi.y-(newh-roi.height)/2,0);
				roi.height = newh;
			}else{
				roi.height = imsize.height-roi.y;
			}
		}

		// get mask of interest motion segment
		if (!maskImage) { maskImage = cvCreateImage(imsize, 8, 1); }
		cvCmpS(imgSegMask, maxloc+1, maskImage, CV_CMP_EQ);

//#if 1
		if ( !m_validator.valid() )                 // waving hand detected !
		{
			err = detect(roi);
			assert(err);
		}else{                                      // tracking waving hand !
			err = myTrack(1);                           // tracking with foreground
			assert(err);
		}
	}else if ( m_validator.valid() ){
		err = myTrack(0);                             // tracking without fgmask
		assert(err);
	}
//#else
//		if ( !m_validator.valid() )                 // waving hand detected !
//		{
//			err = detect(roi);
//			assert(err);
//		}else{                                      // tracking waving hand !
//			err = track(1);                           // tracking with foreground
//			assert(err);
//		}
//	}else if ( m_validator.valid() ){
//		err = track(0);                             // tracking without fgmask
//		assert(err);
//	}
//#endif

	cvReleaseMat(&mask);

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&imgSegMask);
#endif
}

#endif // #if defined(WITH_TZG)
