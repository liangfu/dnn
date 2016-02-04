/**
 * @file   cvchamfer.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jan 14 16:43:27 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvchamfer.h"

void CvChamfer::initialize(const int ntemplates,
                           const CvSize imsize)
{
  m_ntemplates=ntemplates;

  m_original = new CvMat * [ntemplates];
  memset(m_original, 0, sizeof(CvMat*)*ntemplates);

  m_kernel = new CvMat * [ntemplates];
  memset(m_kernel, 0, sizeof(CvMat*)*ntemplates);

  m_center = new CvPoint[ntemplates];
  memset(m_center, 0, sizeof(CvPoint)*ntemplates);

  m_initialized=1;
}

int CvChamfer::load(const char * const fn, const int idx)
{
  if (!m_initialized) {assert(false); return 0;}
  
  assert(m_original[idx]==NULL);
  if (strcmp(fn+strlen(fn)-4, ".xml")==0)
  {
    m_original[idx] = (CvMat*)cvLoad(fn);
  }
  else
  {
    assert(false);
    return 0;
  }

  int nr=m_original[idx]->rows, nc = m_original[idx]->cols;
  // cvThreshold(m_original[idx], m_original[idx], 1, 1, CV_THRESH_BINARY);
  
  m_kernel[idx] = cvCreateMat(nr, nc, CV_32F);
  cvConvert(m_original[idx], m_kernel[idx]);
  m_center[idx]=cvPoint(nr/2, nc/2);
  
  return 1;
}
  
double CvChamfer::weight(const CvMat * distmap,
                         const CvRect roi, const float angle)
{
  int i=0,nr=roi.height,nc=roi.width;
  CvMat subdist_stub, * subdist;
  double retval;
  subdist = cvGetSubRect(distmap, &subdist_stub, roi);
  if (!scoremap)
    scoremap = cvCreateMat(20, 15, CV_32F);

  cvResize(subdist, scoremap);
  cvDiv(scoremap, m_kernel[0], scoremap);
  // cvShowImageEx("Test", scoremap); CV_WAIT();
  // cvShowImageEx("Test", m_kernel[0]); CV_WAIT();
  retval = -log(cvSum(scoremap).val[0]);

  return retval;
}

