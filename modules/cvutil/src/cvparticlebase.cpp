/**
 * @file   cvparticle.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Sun Jan  6 17:35:33 2013
 * 
 * @brief  
 * 
 * 
 */

#include "cvparticlebase.h"
#include "cvparticleutil.h"

/************************* Constructor / Destructor ********************/

/**
 * Allocate Particle filter structure
 *
 * @param num_states    Number of tracking states,
 *                      e.g., 4 if x, y, width, height
 * @param num_particles Number of particles
 * @param logweight     The weights parameter is log  or not
 * @return CvParticle*
 */
CVAPI(CvParticle*) cvCreateParticle( int num_states, 
                                     int num_particles, 
                                     bool logweight  )
{
    CvParticle *p = NULL;
    CV_FUNCNAME( "cvCreateParticle" );
    __BEGIN__;

    CV_ASSERT( num_states > 0 );
    CV_ASSERT( num_particles > 0 );

    p = (CvParticle *) cvAlloc( sizeof( CvParticle ) );
    p->num_particles = num_particles;
    p->num_states    = num_states;
    p->dynamics      = cvCreateMat( num_states, num_states, CV_32FC1 );
    p->rng           = 1;
    p->std           = cvCreateMat( num_states, 1, CV_32FC1 );
    p->bound         = cvCreateMat( num_states, 3, CV_32FC1 );
    p->particles     = cvCreateMat( num_states, num_particles, CV_32FC1 );
    p->weights       = cvCreateMat( 1, num_particles, CV_64FC1 );
    p->logweight     = logweight;
    p->stds          = NULL;

    // Default dynamics: next state = curr state + noise
    cvSetIdentity( p->dynamics, cvScalar(1.0) );
    cvSet( p->std, cvScalar(1.0) );

    cvZero( p->bound );

    __END__;
    return p;
}

/**
 * Release Particle filter structure
 *
 * @param particle
 */
CVAPI(void) cvReleaseParticle( CvParticle** particle )
{
    CvParticle *p = NULL;
    CV_FUNCNAME( "cvReleaseParticle" );
    __BEGIN__;
    p = *particle;
    if( !p ) EXIT;
    
    CV_CALL( cvReleaseMat( &p->dynamics ) );
    CV_CALL( cvReleaseMat( &p->std ) );
    CV_CALL( cvReleaseMat( &p->bound ) );
    CV_CALL( cvReleaseMat( &p->particles ) );
    CV_CALL( cvReleaseMat( &p->weights ) );
    if( p->stds != NULL )
        CV_CALL( cvReleaseMat( &p->stds ) );

    CV_CALL( cvFree( &p ) );
    __END__;
}

/***************************** Setter **********************************/

/**
 * Set dynamics model
 *
 * @param particle
 * @param dynamics (num_states) x (num_states). dynamics model
 *    new_state = dynamics * curr_state + noise
 */
CVAPI(void) cvParticleSetDynamics( CvParticle* p, const CvMat* dynamics )
{
    CV_FUNCNAME( "cvParticleSetDynamics" );
    __BEGIN__;
    CV_ASSERT( p->num_states == dynamics->rows );
    CV_ASSERT( p->num_states == dynamics->cols );
    //cvCopy( dynamics, p->dynamics );
    cvConvert( dynamics, p->dynamics );
    __END__;
}

/**
 * Set noise model
 *
 * @param particle
 * @param rng      random seed. refer cvRNG(time(NULL))
 * @param std      num_states x 1. standard deviation for gaussian noise
 *                 Set standard deviation == 0 for no noise
 */
CVAPI(void) cvParticleSetNoise( CvParticle* p, CvRNG rng, const CvMat* std )
{
    CV_FUNCNAME( "cvParticleSetNoise" );
    __BEGIN__;
    CV_ASSERT( p->num_states == std->rows );
    p->rng = rng;
    //cvCopy( std, p->std );
    cvConvert( std, p->std );
    __END__;
}

/**
 * Set lowerbound and upperbound used for bounding tracking state transition
 *
 * @param particle
 * @param bound    num_states x 3 (lowerbound, upperbound,
 *                                 circular flag 0 or 1)
 *                 Set lowerbound == upperbound to express no bound
 */
CVAPI(void) cvParticleSetBound( CvParticle* p, const CvMat* bound )
{
    CV_FUNCNAME( "cvParticleSetBound" );
    __BEGIN__;
    CV_ASSERT( p->num_states == bound->rows );
    CV_ASSERT( 3 == bound->cols );
    //cvCopy( bound, p->bound );
    cvConvert( bound, p->bound );
    __END__;
}

/************************ Utility *************************************/

/**
 * Get id of the most probable particle
 *
 * @param particle
 * @return int
 */
CVAPI(int) cvParticleGetMax( const CvParticle* p )
{
    double minval, maxval;
    CvPoint min_loc, max_loc;
    cvMinMaxLoc( p->weights, &minval, &maxval, &min_loc, &max_loc );
    return max_loc.x;
}

/**
 * Get the mean state (particle)
 *
 * @param particle
 * @param meanp     num_states x 1, CV_32FC1 or CV_64FC1
 * @return CVAPI(void)
 */
CVAPI(void) cvParticleGetMean( const CvParticle* p, CvMat* meanp )
{
  CvMat* weights = NULL;
  CvMat* particles_i, hdr;
  int i, j;
  CV_FUNCNAME( "cvParticleGetMean" );
  __BEGIN__;

  CV_ASSERT( meanp->rows == p->num_states && meanp->cols == 1 );

  if( !p->logweight )
  {
    weights = p->weights;
  }
  else
  {
    weights = cvCreateMat( 1, p->num_particles, p->weights->type );
    cvExp( p->weights, weights );
  }

  for( i = 0; i < p->num_states; i++ )
  {
    int circular = (int) cvmGet( p->bound, i, 2 );
    if( !circular ) // usual mean
    {
      particles_i = cvGetRow( p->particles, &hdr, i );
      double mean = 0;
      for( j = 0; j < p->num_particles; j++ )
      {
        mean += cvmGet( particles_i, 0, j ) * cvmGet( weights, 0, j );
      }
      cvmSet( meanp, i, 0, mean );
    }
    else // wrapped mean (angle)
    {
      double wrap = cvmGet( p->bound, i, 1 ) - cvmGet( p->bound, i, 0 );
      particles_i = cvGetRow( p->particles, &hdr, i );
      CvScalar mean = cvAngleMean( particles_i, weights, wrap );
      cvmSet( meanp, i, 0, mean.val[0] );
    }
  }

  if( weights != p->weights )
    cvReleaseMat( &weights );

  __END__;
}

/**
 * Print states of a particle
 *
 * @param particle
 * @param p_id      particle id
 */
CVAPI(void) cvParticlePrint( const CvParticle*p, int p_id CV_DEFAULT(-1) )
{
    if( p_id == -1 ) // print all
    {
        int n;
        for( n = 0; n < p->num_particles; n++ )
        {
            cvParticlePrint( p, n );
        }
    }
    else {
        int i;
        for( i = 0; i < p->num_states; i++ )
        {
            printf( "%6.1f ", cvmGet( p->particles, i, p_id ) );
        }
        printf( "\n" );
        fflush( stdout );
    }
}

/****************************** Helper functions **************************/
/*
 * Do normalization of weights
 *
 * @param particle
 * @see cvParticleResample
 */
CVAPI(void) cvParticleNormalize( CvParticle* p )
{
    if( !p->logweight )
    {
        CvScalar normterm = cvSum( p->weights );
        cvScale( p->weights, p->weights, 1.0 / normterm.val[0] );
    }
    else // log version
    {
        CvScalar normterm = cvLogSum( p->weights );
        cvSubS( p->weights, normterm, p->weights );
    }
}

/**
 * Apply lower bound and upper bound for particle states.
 *
 * @param particle
 * @note Used by See also functions
 * @see cvParticleTransition
 */
CVAPI(void) cvParticleBound( CvParticle* p )
{
    int row, col;
    double lower, upper;
    int circular;
    CvMat* stateparticles, hdr;
    float state;
    // @todo: np.width = (double)MAX( 2.0, MIN( maxX - 1 - x, width ) );
    for( row = 0; row < p->num_states; row++ )
    {
        lower = cvmGet( p->bound, row, 0 );
        upper = cvmGet( p->bound, row, 1 );
        circular = (int) cvmGet( p->bound, row, 2 );
        if( lower == upper ) continue; // no bound flag
        if( circular ) {
            for( col = 0; col < p->num_particles; col++ ) {
                state = cvmGet( p->particles, row, col );
                state =
                    ( state <  lower) ? ( state + upper ) :
                    ((state >= upper) ? ( state - upper ) : state );
                cvmSet( p->particles, row, col, state );
            }
        } else {
            stateparticles = cvGetRow( p->particles, &hdr, row );
            cvMinS( stateparticles, upper, stateparticles );
            cvMaxS( stateparticles, lower, stateparticles );
        }
    }
}

/******************* Main (Related to Algorithm) ************************/

/**
 * Initialize states
 *
 * If initial states are given, these states are uniformly copied.
 * If not given, states are uniform randomly sampled within lowerbound 
 * and upperbound regions.
 *
 * @param particle
 * @param init       initial states.
 */
CVAPI(void) cvParticleInit( CvParticle* p, const CvParticle* init = NULL )
{
  int i, c, n, s;

  if ( init ) // copy
  {
    int *num_copy;
    CvMat init_particle;

    int divide = p->num_particles / init->num_particles;
    int remain = p->num_particles - divide * init->num_particles;
    num_copy = (int*) malloc( init->num_particles * sizeof(int) );
    for( i = 0; i < init->num_particles; i++ )
    {
      num_copy[i] = divide + ( i < remain ? 1 : 0 );
    }
        
    n = 0; // copy all states once
    for ( i = 0; i < init->num_particles; i++ )
    {
      cvGetCol( init->particles, &init_particle, i );
      // for ( c = 0; c < num_copy[i]; c++ )
      // {
      //   // cvSetCol( &init_particle, p->particles, n++ );
      //   cvSetCol( &init_particle, p->particles, c );
      // }
      cvRepeat(&init_particle, p->particles);
      // cvDoNothing();
    }

    // randomize partial states if necessary
    for ( s = 0; s < init->num_states; s++ )
    {
      n = 0;
      for ( i = 0; i < init->num_particles; i++ )
      {
        // randomize flag
        if ( FLT_MAX - cvmGet( init->particles, s, i ) < FLT_EPSILON ) 
        {
          CvScalar lower, upper;
          CvMat * statenoiseT =
              cvCreateMat( num_copy[i], 1, p->particles->type );
          lower = cvScalar( cvmGet( p->bound, s, 0 ) );
          upper = cvScalar( cvmGet( p->bound, s, 1 ) );
          cvRandArr( &p->rng, statenoiseT, CV_RAND_UNI, lower, upper );
          
          for( c = 0; c < num_copy[i]; c++ ) {
            cvmSet( p->particles, s, n++, cvmGet( statenoiseT, c, 0 ) );
          }
          cvReleaseMat( &statenoiseT );
        }
      }
    }

    free( num_copy );
  } 
  else // randomize all states
  {
    CvScalar lower, upper;
    CvMat * statenoiseT =
        cvCreateMat( p->num_particles, 1, p->particles->type );
    CvMat * statenoise  =
        cvCreateMat( 1, p->num_particles, p->particles->type );
    
    for (s = 0; s < p->num_states; s++ ) 
    {
      lower = cvScalar( cvmGet( p->bound, s, 0 ) );
      upper = cvScalar( cvmGet( p->bound, s, 1 ) );
      cvRandArr( &p->rng, statenoiseT, CV_RAND_UNI, lower, upper );
      cvT( statenoiseT, statenoise );
      cvSetRow( statenoise, p->particles, s );
    }
    cvReleaseMat( &statenoise );
    cvReleaseMat( &statenoiseT );
  }
}

/**
 * Samples new particles from given particles
 *
 * Currently suppports only linear combination of states transition model. 
 * Write up a function by yourself to supports nonlinear dynamics
 * such as Taylor series model and call your function
 * instead of this function. 
 * Other functions should not necessary be modified.
 *
 * @param particle
 * @note Uses See also functions inside.
 * @see cvParticleBound
 */
CVAPI(void) cvParticleTransition( CvParticle* p )
{
  int i, j;
  CvMat * transits =
      cvCreateMat( p->num_states, p->num_particles, p->particles->type );
  CvMat * noises   =
      cvCreateMat( p->num_states, p->num_particles, p->particles->type );
  CvMat* noise, noisehdr;
  double std;
    
  // dynamics
  cvMatMul( p->dynamics, p->particles, transits );
    
  // noise generation
  if( p->stds == NULL )
  {
    for( i = 0; i < p->num_states; i++ )
    {
      std = cvmGet( p->std, i, 0 );
      noise = cvGetRow( noises, &noisehdr, i );
      if( std == 0.0 )
        cvZero( noise );
      else
        cvRandArr( &p->rng, noise, CV_RAND_NORMAL,
                   cvScalar(0), cvScalar( std ) );
    }
  }
  else
  {
    for( i = 0; i < p->num_states; i++ )
    {
      for( j = 0; j < p->num_particles; j++ )
      {
        std = cvmGet( p->stds, i, j );
        if( std == 0.0 )
          cvmSet( noises, i, j, 0.0 );
        else
          cvmSet( noises, i, j, cvRandGauss( &p->rng, std ) );
      }
    }
  }

  // dynamics + noise
  cvAdd( transits, noises, p->particles );

  cvReleaseMat( &transits );
  cvReleaseMat( &noises );

  cvParticleBound( p );
}

/**
 * Re-samples a set of particles according to their weights to produce a
 * new set of unweighted particles
 *
 * Simply copy, not uniform randomly selects
 *
 * @param particle
 * @note Uses See also functions inside.
 */
CVAPI(void) cvParticleResample( CvParticle* p )
{
  int i, j, np, k = 0;
  CvMat* particle, hdr;
  CvMat* new_particles =
      cvCreateMat( p->num_states, p->num_particles, p->particles->type );
  double weight;
  int max_loc;

  k = 0;
  for( i = 0; i < p->num_particles; i++ )
  {
    particle = cvGetCol( p->particles, &hdr, i );
    weight = cvmGet( p->weights, 0, i );
    weight = p->logweight ? exp( weight ) : weight;
    np = cvRound( weight * p->num_particles );
    for( j = 0; j < np; j++ )
    {
      cvSetCol( particle, new_particles, k++ );
      if( k == p->num_particles ) { goto exit; }
    }
  }

  max_loc = cvParticleGetMax( p );
  particle = cvGetCol( p->particles, &hdr, max_loc );
  while( k < p->num_particles ) {
    cvSetCol( particle, new_particles, k++ );
  }
  
exit:
  cvReleaseMat( &p->particles );
  p->particles = new_particles;
}


