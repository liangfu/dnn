/**
 * @file   main35_face.cpp
 * @author Liangfu Chen <liangfu.chen@cn.fix8.com>
 * @date   Mon Jul  8 16:14:23 2013
 * 
 * @brief  
 * 
 * 
 */
// #include "cvfacedetector.h"
#include "cvext_c.h"
#include "cvtimer.h"
#include "cvimgwarp.h"

typedef struct {char fn[1024];CvBox2D box;} CvTrainSample;
void icvWarp3_8u(CvMat * img, CvMat * dst, CvMat * warp_p);


int main(int argc, char * argv[])
{
  int i,j;
  const char * imglist = "../data/face/imagelist.txt";
  const char * neglist = "../data/face/negative.txt";
  CvTrainSample possamples[1000],negsamples[1000];
  const int maxpossamples = sizeof(possamples)/sizeof(CvTrainSample);
  const int maxnegsamples = sizeof(negsamples)/sizeof(CvTrainSample);
  char line[1024];
  CvPoint2D32f leye,reye,nose,lmou,mmou,rmou;
  float xscale,yscale,scale;
  int poscount=0,negcount=0;

  // load image data
  FILE * fp = fopen(imglist, "r");
  for (i=0;i<maxpossamples;i++){
    fgets(line,1024,fp);
    if (line[0]=='-'){break;}
    if (line[0]=='#'){i--;continue;}
    sscanf(line,"%s %f %f %f %f %f %f %f %f %f %f %f %f",possamples[i].fn,
           &leye.x,&leye.y,&reye.x,&reye.y,&nose.x,&nose.y,
           &lmou.x,&lmou.y,&mmou.x,&mmou.y,&rmou.x,&rmou.y);
    xscale = sqrt(pow(leye.x-reye.x,2)+pow(leye.y-reye.y,2))/12.;
    yscale = sqrt(pow((leye.x+reye.x)*.5-mmou.x,2)+
                  pow((leye.y+reye.y)*.5-mmou.y,2))/12.;
    // fprintf(stderr, "scale: %f,%f\n",xscale,yscale);
    // if (xscale<2.||yscale<2.){i--;continue;}
	scale=(xscale+yscale)*.5;
    possamples[i].box =
	  cvBox2D(nose.x-xscale*1.,
			  // nose.y-yscale*1.,
			  nose.y-yscale*0.,
			  xscale*24.,
			  yscale*24.,
			  -atan2(reye.y-leye.y,reye.x-leye.x)*180./CV_PI);
	  // cvRect(leye.x-6.*scale,leye.y-6.*scale,scale*24.,scale*24.);
	// possamples[i].angle=-atan2(reye.y-leye.y,reye.x-leye.x);
  }
  fclose(fp);
  poscount=i;

  // fp = fopen(neglist,"r");
  // for (i=0;i<maxnegsamples;i++){
  //   fgets(line,1024,fp);
  //   if (line[0]=='-'){break;}
  //   sscanf(line,"%s",negsamples[i].fn);
  // }
  // fclose(fp);
  // negcount=i;
  static CvWindowManager winmgr;
  
  fprintf(stderr, "INFO: %d positive samples!\n", i);

  for (i=0;i<poscount;i++)
  {
	IplImage * tmp = cvLoadImage(possamples[i].fn,0);
	CvBox2D box=possamples[i].box;
	//CvRect roi = cvBox2DToRect(box);
	if (!tmp){fprintf(stderr,"ERROR:file %s not found!\n",possamples[i].fn);break;}
	CvMat * img = cvCreateMat(tmp->height,tmp->width,CV_8U);
	cvCopy(tmp,img);
	CvMat * face = cvCreateMat(24,24,CV_8U);
	//fprintf(stderr,"%d,%d,%d\n",roi.x,roi.y,roi.width);
	CvPoint pts[4]; cvBoxPoints32s(box, pts);
	float warp_data[4]={
	  box.size.width/24.*cos(box.angle*CV_PI/180.)*1.1,
	  -box.size.height/24.*sin(box.angle*CV_PI/180.)*1.1,
	  pts[0].x-box.size.width/24.*1.1*.5,
	  pts[0].y-box.size.height/24.*1.1*.5
	};
	CvMat warp = cvMat(4,1,CV_32F, warp_data);
	icvWarp(img,face,&warp);
	//CV_SHOW(face); 

	char fname[1024];
	sprintf(fname,"../data/face/face24x24/face-%04d.png",i*2);
	cvSaveImage(fname,face);
	cvFlip(face,face,1);
	sprintf(fname,"../data/face/face24x24/face-%04d.png",i*2+1);
	cvSaveImage(fname,face);

	cvReleaseMat(&img);
	cvReleaseMat(&face);
  }
  
  return 0;
}

void icvWarp3_8u(CvMat * img, CvMat * dst, CvMat * warp_p)
{
  assert( CV_MAT_TYPE(warp_p->type)==CV_32F );
  assert( CV_MAT_TYPE(img->type)==CV_8U );           
  assert( CV_MAT_TYPE(dst->type)==CV_8U );           
  assert( (warp_p->cols==1) );                       
  assert( (warp_p->rows==3) );                       
  uchar * iptr = img->data.ptr;                      
  uchar * wptr = dst->data.ptr;                      
  float * pptr = warp_p->data.fl;                    
  int y,x,nr=dst->rows,nc=dst->cols,                 
    istep=img->step/sizeof(uchar),                   
    wstep=dst->step/sizeof(uchar);
  int xpos, ypos;
  float cp0=pptr[0], sp1=0, sp3=0, cp4=pptr[0];
  int xx=1,yy=2;
  int ww=img->width-1, hh=img->height-1;
  for (y=0;y<nr;y++,wptr+=wstep)                             
  {                                                          
    for (x=0;x<nc;x++)                                       
    {                                                        
      xpos = cvRound(x*cp0+y*sp1+pptr[xx]);                  
      ypos = cvRound(x*sp3+y*cp4+pptr[yy]);
	  //fprintf(stderr,"%d,%d\n",xpos,ypos);
      if ( (xpos>ww) || (xpos<0) || (ypos>hh) || (ypos<0) )  
      {                                                      
        wptr[x]=1e-5;                                        
      }else{                                                 
        wptr[x] = (iptr+istep*ypos)[xpos];                   
      }                                                      
    }                                                        
  }                                                         
}

