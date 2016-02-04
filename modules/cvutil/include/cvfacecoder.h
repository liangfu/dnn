/**
 * @file   cvfacecoder.h
 * @author Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>
 * @date   Fri Aug  2 10:47:00 2013
 * 
 * @brief  
 * 
 * 
 */

#ifndef __CV_FACE_CODER_H__
#define __CV_FACE_CODER_H__

#include "cvext_c.h"

// #define WITH_SQLITE3
#ifdef WITH_SQLITE3
#include "cvsql.h"
#endif // WITH_SQLITE3

typedef struct {
  int id;
  char name[32];
  CvMat * data;
  CvMat * img;
} CvFaceRegisterInfoSRC;

class CvFaceCoder
{
 protected:  
  virtual int train(CvMat * img, const char * name)=0;
  virtual double predict(CvMat * img, char * name)=0;

 public:
  CvFaceCoder(){}
  ~CvFaceCoder(){}
};

class CvFaceCoderSRC : public CvFaceCoder
{
  int m_initialized;
  int initialize();

  int M,nsamples,N,maxiter;
  CvMat * D;

  CvFaceRegisterInfoSRC * personlist;
  int nperson;

  int save_db()
  {
    return 1;
  }
  
  int load_db(){
#ifdef WITH_SQLITE3
    release_personlist();
    int i;
    nperson = db.count("test");
    personlist = new CvFaceRegisterInfoSRC[nperson];
    for (i=0;i<nperson;i++){
      personlist[i].id=db.query_int("test",i+1,"id");
    }
    for (i=0;i<nperson;i++){
      personlist[i].data=cvCreateMat(nsamples,maxiter*M,CV_32F);
      personlist[i].img=cvCreateMat(64,56,CV_8U);
      cvZero(personlist[i].data);
      cvZero(personlist[i].img);
      char cond[0xff];sprintf(cond,"where id = %d",personlist[i].id);
db.query("test",cond,"name", 
         personlist[i].name,sizeof(personlist[i].name));
db.query("test",cond,"data", 
         personlist[i].data->data.ptr,sizeof(float)*nsamples*maxiter*M);
db.query("test",cond,"img",
         personlist[i].img->data.ptr,sizeof(uchar)*56*64);
// cvAddS(personlist[i].data,cvScalar(-129.991111),personlist[i].data);
    }
    return 1;
#endif // WITH_SQLITE3
  }

  void release_personlist(){
    int i;
    if (personlist){
      for (i=0;i<nperson;i++){
        cvReleaseMat(&personlist[i].data); personlist[i].data=NULL;
        cvReleaseMat(&personlist[i].img);  personlist[i].img=NULL;
      }
      delete [] personlist; personlist=NULL;
    }else{assert(!nperson);}
  }
  
 protected:
#ifdef WITH_SQLITE3
  CvSql db;
#endif // WITH_SQLITE3
  
 public:
  CvFaceCoderSRC():
      CvFaceCoder(),m_initialized(0),
      M(64),nsamples(56),N(90),maxiter(8),
      D(NULL),personlist(0),nperson(0)
  {
    initialize();
  }
  ~CvFaceCoderSRC(){
    int i;
    if (D){ cvReleaseMat(&D); D=NULL; }

    // release person list
    release_personlist();
  }

  int initialized() {return m_initialized;}
  int config(const char * path);

  int train(CvMat * img, const char * name);
  double predict(CvMat * img, char * name);
};

#endif // __CV_FACE_CODER_H__
