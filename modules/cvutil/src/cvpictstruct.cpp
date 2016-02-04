/**
 * @file   cvpictstruct.cpp
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Jun 28 18:10:57 2013
 * 
 * @brief  
 * 
 * 
 */
#include "cvpictstruct.h"
#include "cvhog.h"
#include "cvimgwarp.h"

double icvPartsMeasure(CvMatND * model, CvMatND * observe){return 0;}

void icvGetWarpFromBox(CvBox2D box, CvSize sz, CvMat * warp_p);

/** 
 * load annotation file for training parts structure
 * 
 * @param trainfile IN:  name of annotation file, end with '--'
 * @param samples   OUT: samples collected within the function
 * 
 * @return OUT: return number of samples correctly collected in the file
 */
int icvLoadPartsSamples(const char * trainfile,
                        CvPartsSampleCollection & samples);

int CvPartsStructure::initialize(CvMat * img, CvBox2D root)
{
  assert(!m_initialized);
  int i,nparts=m_nparts; assert(nparts);
  
  m_particle = new CvParticle*[nparts];
  for (i=0;i<nparts;i++){
    m_particle[i]=cvCreateParticle(4,32,1);
    assert(m_particle[i]->num_states==4);
  }
  assert(m_particle[0]->num_states==4);

  // initialize the root node
#if 0
  float init_particle_data_data[4]={
    root.center.x,root.center.y,1.,root.angle};
  CvMat init_particle_data = cvMat(4,1,CV_32F,init_particle_data_data);
  CvParticle * init_particle = cvCreateParticle(4,1,1);
  assert(m_particle[0]->num_states==4);
  cvCopy(&init_particle_data,init_particle->particles);
  cvParticleInit(m_particle[0],init_particle);
  cvReleaseParticle(&init_particle);
#else
  for (i=0;i<nparts;i++){
    cvParticleInit(m_particle[i],0); // random initialize
  }
#endif

  float dynamics_data[16];
  CvMat dynamics = cvMat(4,4,CV_32F,dynamics_data);
  cvSetIdentity(&dynamics);
  int border = 10;
  float bound_data[12]={
    border,320-border,0,
    border,240-border,0,
    .6,1.67,0,
    -30,30,0
  };
  CvMat bound = cvMat(4,3,CV_32F,bound_data);
  float noise_data[4]={1.0,1.0,0.1,2.0};
  CvMat noise = cvMat(4,1,CV_32F,noise_data);
  CvRNG rng = cvRNG();
  for (i=0;i<nparts;i++){
    cvParticleSetDynamics(m_particle[i],&dynamics);
    cvParticleSetBound(m_particle[i],&bound);
    cvParticleSetNoise(m_particle[i],rng,&noise);
  }

  m_initialized=1;
  return 1;
}

int CvPartsStructure::update(CvMat * img)
{
  int i,nparts=m_nparts;
  for (i=0;i<nparts;i++){
    cvParticleTransition(m_particle[i]);
  }

  for (i=0;i<nparts;i++){
    //icvPartsMeasure();
  }

  for (i=0;i<nparts;i++){
    cvParticleNormalize(m_particle[i]);
  }
  for (i=0;i<nparts;i++){
    cvParticleResample(m_particle[i]);
  }
  return 0;
}

int CvPartsStructure::train(char * trainfile)
{
  int i,j;
  int retval=1;

  icvLoadPartsSamples(trainfile, m_data);
  int nparts = m_data.nparts;
  m_nparts=nparts;
  int nsamples = m_data.nsamples;
  m_data.features = new CvMatND*[nparts];
  for (i=0;i<nparts;i++){
    int hogsizes[3]={
      cvFloor(m_data.tsizes[i].h/6.),
      cvFloor(m_data.tsizes[i].w/6.),9.};
    m_data.features[i] = cvCreateMatND(3,hogsizes,CV_32F);
  }

  /// display annotations
  for (i=0;i<nsamples;i++)
  {
    IplImage * tmp = cvLoadImage(m_data.samples[i].fn,1);
    CvMat img_stub;
    CvMat * img = cvGetMat(tmp,&img_stub);
    for (j=0;j<nparts;j++){
      CvPart & p = m_data.samples[i].d[j];
      CvBox2D box = cvBox2D(p.x,p.y,p.w,p.h,p.r*57.29578);
      cvBoxRectangle(img,box,CV_RED);
      cvCircle(img,cvPoint(p.x,p.y),2,CV_GREEN);
    }
    cvShowImage("Test",img); CV_WAIT();
  }

  /// TODO: train each individual part, get parts connection info
  for (i=0;i<nsamples;i++)
  {
    IplImage * raw = cvLoadImage(m_data.samples[i].fn,0);
    CvMat img_stub;
    CvMat * img = cvGetMat(raw, &img_stub);

    for (j=0;j<nparts;j++){
      CvPart & p = m_data.samples[i].d[j];
      CvBox2D box = cvBox2D(p.x,p.y,p.w,p.h,p.r*57.29578);
      float warp_p_data[4];
      CvMat warp_p = cvMat(4,1,CV_32F,warp_p_data);
      icvGetWarpFromBox(box,cvSize(m_data.tsizes[j].w,
                                   m_data.tsizes[j].h), &warp_p);
      cvPrintf(stderr, "%f,", &warp_p);
      CvMat * patch =
          cvCreateMat(m_data.tsizes[j].h,m_data.tsizes[j].w,CV_8U);
      icvWarp(img,patch,&warp_p);
      CV_SHOW(patch);
      icvCalcHOG(patch,m_data.features[j]);
      icvShowHOG(m_data.features[j],CV_CM_GRAY,2); CV_WAIT();
      cvReleaseMat(&patch);
    }
  }
  
  return retval;
}

int icvLoadPartsSamples(const char * trainfile,
                        CvPartsSampleCollection & data)
{
  int i,j,retval=1;
  char line[1024];
  int nsamples=0,nparts=0;
  FILE * fp = fopen(trainfile,"r");
  if (!fp) {
    fprintf(stderr,"ERROR: file %s not found", trainfile);return 0;
  }

  fgets(line,1024,fp);
  sscanf(line,"%d",&nparts);

  CvPartsSample samples[100];
  // read annotations
  for (i=0;;i++)
  {
    fgets(line,1024,fp);
    if (line[0]=='#'){i--;continue;}
    if (line[0]=='!'){
      char varname[64];
      sscanf(line,"%s",varname);
      if (strcmp(varname+1,"tsizes")!=0){
        fprintf(stderr,"ERROR: unknown variable name '%s'\n",varname+1);
        retval=0; break;
      }
      for (j=0;j<nparts;j++){
        fgets(line,1024,fp);
        sscanf(line,"%d %d",&data.tsizes[j].w,&data.tsizes[j].h);
      }
      i--;continue;
    }
    if (line[0]=='-'){break;}
    sscanf(line,"%s",samples[i].fn);
    //samples[i].parts=nparts;
    for (j=0;j<nparts;j++){
      fgets(line,1024,fp);
      sscanf(line,"%f %f %f %f %f",
             &samples[i].d[j].x,&samples[i].d[j].y,
             &samples[i].d[j].w,&samples[i].d[j].h,
             &samples[i].d[j].r);
      samples[i].d[j].r=samples[i].d[j].r*CV_PI/180.;
    }
  }
  nsamples = i;

  data.nparts = nparts;
  data.nsamples = nsamples;
  assert(!data.samples);
  data.samples = new CvPartsSample[nsamples];
  memcpy(data.samples,samples,nsamples*sizeof(CvPartsSample));

  fclose(fp);
  return retval;
}

void icvGetWarpFromBox(CvBox2D box, CvSize sz, CvMat * warp_p)
{
  CvPoint2D32f center = box.center;
  float halfw = float(sz.width)*.5;
  float halfh = float(sz.height)*.5;
  float xscale = box.size.width/float(sz.width);
  float yscale = box.size.height/float(sz.height);
  float theta = -box.angle*CV_PI/180.;
  float costheta = cos(theta);
  float sintheta = sin(theta);

  assert(warp_p->rows==4);
  assert(CV_MAT_TYPE(warp_p->type)==CV_32F);
  float warp_p_data[4] = {
    xscale*costheta,yscale*sintheta,
    center.x-halfw*xscale*costheta+halfh*yscale*sintheta,
    center.y-halfw*xscale*sintheta-halfh*yscale*costheta
  };
  memcpy(warp_p->data.fl,warp_p_data,sizeof(warp_p_data));
}
