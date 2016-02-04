/**
 * @file   cvpictstruct.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Jun 28 17:52:23 2013
 * 
 * @brief  
 * 
 * 
 */
#ifndef __CV_PICTORIAL_STRUTURE_H__
#define __CV_PICTORIAL_STRUTURE_H__

#include "cvext_c.h"
#include "cvparticlebase.h"


#define CV_MAX_PARTS 10
typedef struct { float x,y,w,h,r; } CvPart;
typedef struct { char fn[256]; CvPart d[CV_MAX_PARTS]; } CvPartsSample;
typedef struct {
  struct {int w,h;} tsizes[CV_MAX_PARTS];
  int nparts;
  int nsamples;
  CvPartsSample * samples;
  CvMatND ** features;
} CvPartsSampleCollection;

class CV_EXPORTS CvPartsStructure
{
  int m_initialized;

  int m_nparts;
  CvMat * m_parts_tsize;
  CvMat * m_parts_location;
  CvMat * m_parts_appearance;

  CvParticle ** m_particle;
  CvMat * m_configuration;

  CvPartsSampleCollection m_data;
  
protected:
  // int m_nparts;
  // CvMat ** parts_mean; // statistics of parts
  // CvMat ** parts_covar;

  // int num_nodes;
  // CvMat * nodes; // location
  // int num_edges;
  // CvMat * edges; // connection of nodes

public:
  CvPartsStructure():
      m_initialized(false),m_nparts(0),
      m_parts_location(NULL),m_parts_appearance(NULL),
      m_particle(NULL),m_configuration(NULL)
      // m_nparts(-1),parts_mean(NULL),parts_covar(NULL),
      // num_nodes(-1),num_edges(-1),
      // nodes(NULL),edges(NULL)
  {
    m_data.samples = NULL;
    m_data.features = NULL;
  }

  ~CvPartsStructure()
  {
    int i;

    // release data sample collection
    if (m_data.samples) {
      delete [] m_data.samples;
    }
    if (m_data.features){
      for (i=0;i<m_nparts;i++){
        if (m_data.features[i]) { cvReleaseMatND(&m_data.features[i]); }
      }
      delete [] m_data.features;
    }
    
    if (m_particle){
      for (i=0;i<m_nparts;i++){
        cvReleaseParticle(&m_particle[i]);
      }
    }
    if (m_configuration) {
      cvReleaseMat(&m_configuration); m_configuration=NULL;
    }
  }

  int initialized() { return m_initialized; }
  int initialize(CvMat * img, CvBox2D root);
  int update(CvMat * img);
  CvMat * configuration(){ return m_configuration; }
  
  int train(char * trainfilename);
};

#endif // __CV_PICTORIAL_STRUTURE_H__
