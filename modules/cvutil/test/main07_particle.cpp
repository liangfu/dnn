#include "cvext.h"
#include "cvextparticle.h"


/****************************** Global *****************************/

int num_particles = 100;
// state.h
extern int num_states;
// observation.h
const char* data_pcaval = "pcaval.xml";
const char* data_pcavec = "pcavec.xml";
const char* data_pcaavg = "pcaavg.xml";
CvSize featsize = cvSize(0,0);

bool arg_export = false;
char export_filename[2048];
const char* export_format = "%s_%04d.png";

float std_x = 3.0;
float std_y = 3.0;
float std_w = 2.0;
float std_h = 2.0;
float std_r = 1.0;

const char* vid_file = "02291vFF.avi";

/******************************* Structures ************************/

typedef struct IcvMouseParam {
    CvPoint loc1;
    CvPoint loc2;
    char* win_name;
    IplImage* frame;
} IcvMouseParam;

/**************************** Function Prototypes ******************/

void icvGetRegion( IplImage*, CvRect* );
void icvMouseCallback( int, int, int, int, void* );
void usage( );
void arg_parse( int argc, char** argv );
/**************************** Main ********************************/
int main( int argc, char** argv )
{
  IplImage *frame;
  CvCapture* video;
  int frame_num = 0;
  int i;

  arg_parse( argc, argv );

  // load trained PCA data
  cvParticleObserveInitialize( data_pcaval, data_pcavec, data_pcaavg, featsize );

  // read a video
  if( !vid_file || (isdigit(vid_file[0]) && vid_file[1] == '\0') )
    video = cvCaptureFromCAM( !vid_file ? 0 : vid_file[0] - '0' );
  else
    video = cvCaptureFromAVI( vid_file );  
  if( (frame = cvQueryFrame( video )) == NULL )
  {
    fprintf( stderr, "Video %s is not loadable.\n", vid_file );
    usage();
    exit(1);
  }

  // allows user to select initial region
  CvRect region;
  icvGetRegion( frame, &region );

  // configure particle filter
  bool logprob = true;
  CvParticle *particle = cvCreateParticle( num_states, num_particles, logprob );
  CvParticleState std = cvParticleState (
      std_x,
      std_y,
      std_w,
      std_h,
      std_r
                                         );
  cvParticleStateConfig( particle, cvGetSize(frame), std );

  // initialize particle filter
  CvParticleState s;
  CvParticle *init_particle;
  init_particle = cvCreateParticle( num_states, 1 );
  CvRect32f region32f = cvRect32fFromRect( region );
  CvBox32f box = cvBox32fFromRect32f( region32f ); // centerize
  s = cvParticleState( box.cx, box.cy, box.width, box.height, 0.0 );
  cvParticleStateSet( init_particle, 0, s );
  cvParticleInit( particle, init_particle );
  cvReleaseParticle( &init_particle );

  while( ( frame = cvQueryFrame( video ) ) != NULL )
  {
    // Draw new particles
    cvParticleTransition( particle );
    // Measurement
    cvParticleObserveMeasure( particle, frame );

    // Draw all particles
    for( i = 0; i < particle->num_particles; i++ )
    {
      CvParticleState s = cvParticleStateGet( particle, i );
      cvParticleStateDisplay( s, frame, CV_RGB(0,0,255) );
    }
    // Draw most probable particle
    //printf( "Most probable particle's state\n" );
    int maxp_id = cvParticleGetMax( particle );
    CvParticleState maxs = cvParticleStateGet( particle, maxp_id );
    cvParticleStateDisplay( maxs, frame, CV_RGB(255,0,0) );
    ///cvParticleStatePrint( maxs );
        
    // Save pictures
    if( arg_export ) {
      sprintf( export_filename, export_format, vid_file, frame_num );
      printf( "Export: %s\n", export_filename ); fflush( stdout );
      cvSaveImage( export_filename, frame );
    }
    cvShowImage( "Select an initial region > SPACE > ESC", frame );

    // Normalize
    cvParticleNormalize( particle);
    // Resampling
    cvParticleResample( particle );

    char c = cvWaitKey( 1000 );
    if(c == '\x1b')
      break;
  }

  cvParticleObserveFinalize();
  cvDestroyWindow( "Select an initial region > SPACE > ESC" );
  cvReleaseParticle( &particle );
  cvReleaseCapture( &video );
  return 0;
}

/**
 * Print usage
 */
void usage()
{
    fprintf( stderr,
             "Object tracking using particle filtering with PCA DIFS + DFFS method.\n"
             "Usage: facetrack\n"
             " --pcaval <pcaval_xml_path = %s>\n"
             "     Specify trained PCA model data.\n"
             " --pcavec <pcavec_xml_path = %s>\n"
             "     Specify trained PCA model data.\n"
             " --pcaavg <pcaavg_xml_path = %s>\n"
             "     Specify trained PCA model data.\n"
             " --featsize <width> <height> (Default: (int)sqrt(pcaavg->rows) both)\n"
             "     Preprocessing image size. \n"
             " -p <num_particles = %d>\n"
             "     Number of particles (generated tracking states). \n"
             " -sx <noise_std_for_x = %f>\n"
             "     The standard deviation sigma of the Gaussian window for the x coord. \n"
             "     The Gaussian distribution has a good property called 68-95-99.7 rule. \n"
             "     Intuitively, the searching window range is +-2sigma in 95%%. \n"
             " -sy <noise_std_for_y = %f>\n"
             "     sigma for image patch y coord.\n"
             " -sw <noise_std_for_width = %f>\n"
             "     sigma for image patch width.\n"
             " -sh <noise_std_for_height = %f>\n"
             "     sigma for image patch height.\n"
             " -sr <noise_std_for_rotate = %f>\n"
             "     sigma for image patch rotation angle in degree.\n"
             " -o\n"
             "     Export resulting frames\n"
             " --export_format <export_format = %s>\n"
             "     Determine the exported filenames as sprintf. \n"
             " -h\n"
             "     Show this help\n"
             " <vid_file = %s | camera_index>\n",
             data_pcaval, data_pcavec, data_pcaavg,
             num_particles,
             std_x, std_y, std_w, std_h, std_r,
             export_format, vid_file );
}

/**
 * Parse command arguments
 */
void arg_parse( int argc, char** argv )
{
    int i;
    for( i = 1; i < argc; i++ )
    {
        if( !strcmp( argv[i], "-h" ) )
        {
            usage();
            exit(0);
        }
        else if( !strcmp( argv[i], "--pcaval" ) )
        {
            data_pcaval = argv[++i];
        }
        else if( !strcmp( argv[i], "--pcavec" ) )
        {
            data_pcavec = argv[++i];
        }
        else if( !strcmp( argv[i], "--pcaavg" ) )
        {
            data_pcaavg = argv[++i];
        }
        else if( !strcmp( argv[i], "--featsize" ) )
        {
            int width = atoi( argv[++i] );
            int height = atoi( argv[++i] );
            featsize = cvSize( width, height );
        }
        else if( !strcmp( argv[i], "-o" ) )
        {
            arg_export = true;
        }
        else if( !strcmp( argv[i], "--export_format" ) )
        {
            export_format = argv[++i];
        }
        else if( !strcmp( argv[i], "-p" ) )
        {
            num_particles = atoi( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sx" ) )
        {
            std_x = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sy" ) )
        {
            std_y = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sw" ) )
        {
            std_w = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sh" ) )
        {
            std_h = atof( argv[++i] );
        }
        else if( !strcmp( argv[i], "-sr" ) )
        {
            std_r = atof( argv[++i] );
        }
        else 
        {
            vid_file = argv[i];
        }
    }
}

/**
 * Allows the user to interactively select the initial object region
 *
 * @param frame  The frame of video in which objects are to be selected
 * @param region A pointer to an array to be filled with rectangles
 */
void icvGetRegion( IplImage* frame, CvRect* region )
{
    IcvMouseParam p;

    /* use mouse callback to allow user to define object regions */
    p.win_name = "Select an initial region > SPACE > ESC";
    p.frame = frame;

    cvNamedWindow( p.win_name, 1 );
    cvShowImage( p.win_name, frame );
    cvSetMouseCallback( p.win_name, &icvMouseCallback, &p );
    cvWaitKey( 0 );
    //cvDestroyWindow( win_name );

    /* extract regions defined by user; store as a rectangle */
    region->x      = min( p.loc1.x, p.loc2.x );
    region->y      = min( p.loc1.y, p.loc2.y );
    region->width  = max( p.loc1.x, p.loc2.x ) - region->x + 1;
    region->height = max( p.loc1.y, p.loc2.y ) - region->y + 1;
}

/**
 * Mouse callback function that allows user to specify the 
 * initial object region. 
 * Parameters are as specified in OpenCV documentation.
 */
void icvMouseCallback( int event, int x, int y, int flags, void* param )
{
    IcvMouseParam* p = (IcvMouseParam*)param;
    IplImage* clone;
    static int pressed = false;

    /* on left button press, remember first corner of rectangle around object */
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        p->loc1.x = x;
        p->loc1.y = y;
        pressed = true;
    }

    /* on left button up, finalize the rectangle and draw it */
    else if( event == CV_EVENT_LBUTTONUP )
    {
        p->loc2.x = x;
        p->loc2.y = y;
        clone = (IplImage*)cvClone( p->frame );
        cvRectangle( clone, p->loc1, p->loc2, CV_RGB(255,255,255), 1, 8, 0 );
        cvShowImage( p->win_name, clone );
        cvReleaseImage( &clone );
        pressed = false;
    }

    /* on mouse move with left button down, draw rectangle */
    else if( event == CV_EVENT_MOUSEMOVE  &&  flags & CV_EVENT_FLAG_LBUTTON )
    {
        clone = (IplImage*)cvClone( p->frame );
        cvRectangle( clone, p->loc1, cvPoint(x, y), CV_RGB(255,255,255), 1, 8, 0 );
        cvShowImage( p->win_name, clone );
        cvReleaseImage( &clone );
    }
}
