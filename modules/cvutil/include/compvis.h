/**
 * @file   compvis.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Tue Dec 31 09:54:18 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __COMP_VIS_H__
#define __COMP_VIS_H__

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <memory.h>
#include <ctype.h>

#include "cxcore.h"

//-------------------------------------------------------
// interfaces
//-------------------------------------------------------

void normalize(float v[3]);
void ncrossprod(float v1[3], float v2[3], float cp[3]);
void triagnormal(float v1[3], float v2[3], float v3[3], float norm[3]);
void cvCalcSurfaceNormal(CvMat * verts, CvMat * faces, CvMat * norms);
int cvLoadSurface(const char * fn, CvMat ** verts, CvMat ** faces);

//-------------------------------------------------------
// implementations
//-------------------------------------------------------

// normalizes v
CV_INLINE
void normalize(float v[3])
{
  float d = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (d == 0){fprintf(stderr, "Zero length vector in normalize\n");}
  else{v[0] /= d; v[1] /= d; v[2] /= d;}
}

// calculates a normalized crossproduct to v1, v2 
CV_INLINE
void ncrossprod(float v1[3], float v2[3], float cp[3])
{
  cp[0] = v1[1]*v2[2] - v1[2]*v2[1];
  cp[1] = v1[2]*v2[0] - v1[0]*v2[2];
  cp[2] = v1[0]*v2[1] - v1[1]*v2[0];
  normalize(cp);
}

// calculates normal to the triangle designated by v1, v2, v3 
CV_INLINE
void triagnormal(float v1[3], float v2[3], float v3[3], float norm[3])
{
  float vec1[3], vec2[3];
  vec1[0] = v3[0] - v1[0];  vec2[0] = v2[0] - v1[0];
  vec1[1] = v3[1] - v1[1];  vec2[1] = v2[1] - v1[1];
  vec1[2] = v3[2] - v1[2];  vec2[2] = v2[2] - v1[2];
  ncrossprod(vec2, vec1, norm);
}

// calculate normal vector for each triangle
#define CV_SHADING_FLAT   0
#define CV_SHADING_SMOOTH 1
void cvCalcSurfaceNormal_flat(CvMat * verts, CvMat * faces, CvMat * norms);
void cvCalcSurfaceNormal_smooth(CvMat * verts, CvMat * faces, CvMat * norms);

CV_INLINE
void cvCalcSurfaceNormal(CvMat * verts, CvMat * faces, CvMat * norms, int mode=0)
{
  typedef void (*CvCalcSurfaceNormalFunc)(CvMat * verts, CvMat * faces, CvMat * norms);
  CvCalcSurfaceNormalFunc func[2]={
	cvCalcSurfaceNormal_flat,
	cvCalcSurfaceNormal_smooth
  };
  assert((mode==CV_SHADING_FLAT)||(mode==CV_SHADING_SMOOTH));
  func[mode](verts,faces,norms);
}

CV_INLINE
void cvCalcSurfaceNormal_flat(CvMat * verts, CvMat * faces, CvMat * norms)
{
  int i;
  int nverts=verts->rows;
  int nfaces=faces->rows;
  assert(verts->cols==3);
  assert(faces->cols==3);
  assert(norms->cols==3);
  assert(faces->rows==norms->rows);
  assert((CV_MAT_TYPE(verts->type)==CV_32F)&&
		 (CV_MAT_TYPE(faces->type)==CV_32S)&&
		 (CV_MAT_TYPE(norms->type)==CV_32F));
  for (i=0;i<nfaces;i++){
	int * fptr=faces->data.i;
	float v1[3]={
	  (verts->data.fl+3*((fptr+3*i)[0]))[0],
	  (verts->data.fl+3*((fptr+3*i)[0]))[1],
	  (verts->data.fl+3*((fptr+3*i)[0]))[2]
	};
	float v2[3]={
	  (verts->data.fl+3*((fptr+3*i)[1]))[0],
	  (verts->data.fl+3*((fptr+3*i)[1]))[1],
	  (verts->data.fl+3*((fptr+3*i)[1]))[2]
	};
	float v3[3]={
	  (verts->data.fl+3*((fptr+3*i)[2]))[0],
	  (verts->data.fl+3*((fptr+3*i)[2]))[1],
	  (verts->data.fl+3*((fptr+3*i)[2]))[2]
	};
	triagnormal(v1,v2,v3,norms->data.fl+3*i);
  }
}

CV_INLINE
void cvCalcSurfaceNormal_smooth(CvMat * verts, CvMat * faces, CvMat * norms)
{
  int i,j,k;
  int nverts=verts->rows;
  int nfaces=faces->rows;
  assert(verts->cols==3);
  assert(faces->cols==3);
  assert(norms->cols==3);
  assert(verts->rows==norms->rows);
  assert((CV_MAT_TYPE(verts->type)==CV_32F)&&
		 (CV_MAT_TYPE(faces->type)==CV_32S)&&
		 (CV_MAT_TYPE(norms->type)==CV_32F));
  memset(norms->data.ptr,0,norms->rows*norms->step);
  float * vptr=verts->data.fl;
  float * nptr=norms->data.fl;
  for (i=0;i<nfaces;i++){
	int * fptr=faces->data.i;
	int idx[3]={(fptr+3*i)[0],(fptr+3*i)[1],(fptr+3*i)[2]};
	float v1[3]={(vptr+3*idx[0])[0],(vptr+3*idx[0])[1],(vptr+3*idx[0])[2]};
	float v2[3]={(vptr+3*idx[1])[0],(vptr+3*idx[1])[1],(vptr+3*idx[1])[2]};
	float v3[3]={(vptr+3*idx[2])[0],(vptr+3*idx[2])[1],(vptr+3*idx[2])[2]};
	float n[3];
	triagnormal(v1,v2,v3,n);
	for (j=0;j<3;j++){for (k=0;k<3;k++){(nptr+3*(idx[j]))[k]+=n[k];}}
  }
  for (i=0;i<nverts;i++){
	normalize(norms->data.fl+3*i);
  }
}

CV_INLINE
int cvLoadSurface(const char * fn, CvMat ** verts, CvMat ** faces)
{
  static const int MAXLEN=1024;
  if (strcmp(fn+strlen(fn)-4,".obj")){
	fprintf(stderr,"ERROR: not wavefont .OBJ file\n",fn);return 0;
  }
  FILE * fp = fopen(fn,"rt");
  if (!fp){fprintf(stderr,"ERROR: file '%s' not exist!\n",fn);return 0;}
  char line[MAXLEN];

  int nverts=0,nfaces=0;
  while(1)
  {
	fgets(line,MAXLEN,fp);
	if (feof(fp)){break;}
	// fprintf(stderr,"%s",line);
	if (line[0]=='#'){continue;}
	else if (!strncmp("v ",line,2)){nverts++;}
	else if (!strncmp("f ",line,2)){nfaces++;}
  }
  fclose(fp);

  *verts = cvCreateMat(nverts,3,CV_32F);
  *faces = cvCreateMat(nfaces,3,CV_32S);

  int i;
  int viter=0,fiter=0;
  float v[3];
  int f[3];
  fp = fopen(fn,"rt");
  if (!fp){fprintf(stderr,"ERROR: file '%s' not exist!\n",fn);return 0;}
  while(1)
  {
	fgets(line,MAXLEN,fp);
	if (feof(fp)){break;}
	// fprintf(stderr,"%s",line);
	if (line[0]=='#'){
	  // ...
	}else if (!strncmp("v ",line,2)){
	  sscanf(line+2,"%f %f %f",&v[0],&v[1],&v[2]);
	  memcpy((*verts)->data.fl+3*viter,v,sizeof(v));
	  viter++;
	}else if (!strncmp("f ",line,2)){
	  sscanf(line+2,"%d %d %d",&f[0],&f[1],&f[2]);
	  for (i=0;i<3;i++){f[i]-=1;}
	  memcpy((*faces)->data.i+3*fiter,f,sizeof(f));
	  fiter++;
	}else if (line[0]=='\n'){
	}else{
	  assert(false);
	}
  }
  assert(viter==nverts);
  assert(fiter==nfaces); // ensure all vertices and faces data are collected
  fclose(fp);
  return 1;
}

CV_INLINE
int cvGetFileSuffix(const char * fullname, char * suffix)
{
  int i,retval;
  int len=strlen(fullname);
  for (i=len-1;i>0;i--){
	if (fullname[i]=='.'){retval=i;break;}
  }
  strncpy(suffix,fullname+retval,len-retval);
  return retval;
}

#endif // __COMP_VIS_H__
