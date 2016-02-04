/**
 * @file   cvparticlestate.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Mon Jun 24 11:32:06 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_PARTICLE_STATE_H__
#define __CV_PARTICLE_STATE_H__

#include "cvext_c.h"
#include "cvparticlebase.h"

void icvParticleStateWarpConfigure(CvParticle * p,
                                   CvSize imsize, CvMat * warp_p);
void icvParticleStateWarpSet(CvParticle * p, int idx, CvMat * warp_p);
void icvParticleStateWarpGet(CvParticle * p, int idx, CvMat * warp_p);

#endif // __CV_PARTICLE_STATE_H__
