/** @file
 * Particle Filter
 *
 * Currently suppports only linear state transition model. 
 * Write up a function by yourself to supports nonlinear dynamics
 * such as Taylor series model and call your function instead of 
 * cvParticleTransition( p ). 
 * Other functions should not necessary be modified.
 *
 * cvCreateParticle
 * cvPartcileSetXxx
 * cvParticleInit
 * loop { 
 *   cvParticleTransition
 *   Measurement
 *   cvParticleNormalize
 *   cvParticleResample
 * }
 * cvReleaseParticle
 */
/* The MIT License
 * 
 * Copyright (c) 2008, Naotoshi Seo <sonots(at)sonots.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef __CV_PARTICLE_INCLUDED__
#define __CV_PARTICLE_INCLUDED__

#include "cvext_c.h"
// #include "cvaux.h"
// #include "cv.h"
// #include "cxcore.h"

#include <time.h>
// #include "cvsetrow.h"
// #include "cvsetcol.h"
// #include "cvlogsum.h"
// #include "cvanglemean.h"
// #include "cvrandgauss.h"

/******************************* Structures *******************************/
/**
 * Particle Filter structure
 */
typedef struct CvParticle {
  // config
  int num_states;    /**< Number of tracking states, e.g.,
                        4 if x, y, width, height */
  int num_particles; /**< Number of particles */
  bool logweight;    /**< log weights are stored in "weights". */
  // transition
  CvMat* dynamics;   /**< num_states x num_states. Dynamics model. */
  CvRNG  rng;        /**< Random seed */
  CvMat* std;        /**< num_states x 1.
                        Standard deviation for gaussian noise
                        Set standard deviation == 0 for no noise */
  CvMat* stds;       /**< num_states x num_particles. 
                        Std for each particle so that you could be varying 
                        noise variance for each particle.
                        "std" is used if "stds" is not set. */
  CvMat* bound;      /**< num_states x 3 (lowerbound, upperbound, 
                        wrap_around (like angle) flag 0 or 1)
                        Set lowerbound == upperbound to express no bound */
  // particle states
  CvMat* particles;  /**< num_states x num_particles. The particles. 
                        The transition states values of all particles. */
  CvMat* weights;    /**< 1 x num_particles. The weights of 
                        each particle respect to the particle id in
                        "particles". "weights" are used to approximated
                        the posterior pdf. */
} CvParticle;

/**************************** Function Prototypes ************************/

CVAPI(CvParticle*) cvCreateParticle( int num_states, int num_particles,
                                     bool logweight CV_DEFAULT(false) );
CVAPI(void) cvReleaseParticle( CvParticle** p );

CVAPI(void) cvParticleSetDynamics( CvParticle* p, const CvMat* dynamics );
CVAPI(void) cvParticleSetNoise( CvParticle* p, CvRNG rng, const CvMat* std );
CVAPI(void) cvParticleSetBound( CvParticle* p, const CvMat* bound );

CVAPI(int)  cvParticleGetMax( const CvParticle* p );
CVAPI(void) cvParticleGetMean( const CvParticle* p, CvMat* meanp );
CVAPI(void) cvParticlePrint( const CvParticle* p, int p_id );

CVAPI(void) cvParticleBound( CvParticle* p );
CVAPI(void) cvParticleNormalize( CvParticle* p );

CVAPI(void) cvParticleInit( CvParticle* p, const CvParticle* init );
CVAPI(void) cvParticleTransition( CvParticle* p );
CVAPI(void) cvParticleResample( CvParticle* p );


#endif // __CV_PARTICLE_INCLUDED__
