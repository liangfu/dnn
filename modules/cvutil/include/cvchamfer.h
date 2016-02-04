/**
 * @file   cvchamfer.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jan 14 16:38:12 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_EXT_CHAMFER_H__
#define __CV_EXT_CHAMFER_H__

#include "cvext_c.h"
// #include "cvaux.hpp"

class CV_EXPORTS CvChamfer
{
  int m_ntemplates;
  int m_initialized;

public:
  CvMat ** m_original;
  CvMat ** m_kernel;
  CvPoint * m_center;
  CvMat * scoremap;

  CvChamfer():
      m_original(NULL), m_kernel(NULL), m_center(NULL), scoremap(NULL),
      m_ntemplates(0), m_initialized(0)
  {
  }

  ~CvChamfer()
  {
    int i;

    if (m_original)
    {
      for (i=0; i < m_ntemplates; i++)
        if (m_original[i]!=NULL)
        {
          cvReleaseMat(&m_original[i]);
          m_original[i]=NULL;
        }
      m_original=NULL;
    }

    if (m_kernel)
    {
      for (i=0; i < m_ntemplates; i++)
        if (m_kernel[i]!=NULL)
        {
          cvReleaseMat(&m_kernel[i]);
          m_kernel[i]=NULL;
        }
      m_kernel=NULL;
    }

    if (m_center)
    {
      delete [] m_center;
      m_center=NULL;
    }

    if (scoremap)
    {
      cvReleaseMat(&scoremap);
      scoremap=NULL;
    }
  }

  /** 
   * allocate memory for list of templates
   * 
   * @param ntemplates 
   */
  void initialize(const int ntemplates, const CvSize imsize);
  int initialized() { return m_initialized; }

  /** 
   * assuming number of templates initialized!
   * 
   * load .XML files as template images with index in the list of templates
   * 
   * @param fn      in:  the .XML file name
   * @param idx     in:  the index in list of templates
   * 
   * @return 0-failure, 1-succcess
   */
  int load(const char * const fn, const int idx);

  /**
   * assuming all templates loaded
   * 
   * calculate chamfer distance with templates,
   * returning matching score and template id
   * 
   * @param bw      in: input binary image for searching the target
   * @param score   out: matching score, normalized with area of template
   * @param id      out: highest matching template id among preloaded ones
   */
  // void search(const CvMat * bw, float & score, int & id);
  double weight(const CvMat * distmap,
                const CvRect roi, const float angle);

  /** 
   * generate binary image from magnitude of gradient images
   * for calculating chamfer distance. 
   * 
   * @param mag    in:  grayscale image
   * @param bw     out: binary image for searching step !
   * @param thres  in:  thresholding value for magnitude of gradient images
   */
  static void prepare(const CvMat * mag, CvMat * bw, const float thres)
  {
    cvThreshold(mag, bw, thres, 1, CV_THRESH_BINARY);
  }
};

#endif // __CV_EXT_CHAMFER_H__
