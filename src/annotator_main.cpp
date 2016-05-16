/** -*- c++ -*- 
 *
 * \file   annotator_main.cpp
 * \date   Mon May  9 13:07:02 2016
 *
 * \copyright 
 * Copyright (c) 2016 Liangfu Chen <liangfu.chen@nlpr.ia.ac.cn>.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms are permitted
 * provided that the above copyright notice and this paragraph are
 * duplicated in all such forms and that any documentation,
 * advertising materials, and other materials related to such
 * distribution and use acknowledge that the software was developed
 * by the Brainnetome Center & NLPR at Institute of Automation, CAS. The 
 * name of the Brainnetome Center & NLPR at Institute of Automation, CAS 
 * may not be used to endorse or promote products derived
 * from this software without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * 
 * \brief  
 */

#include "ml.h"
#include "highgui.h"
#include "cxcore.h"
#include "cvext.h"
#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;
typedef struct CvMouseInfo {
  List<CvPoint> centers;
  int token;
  IplImage * img;
  char imgname[1<<10];
  CvMouseInfo():token(0),img(0){}
  void clear(){centers.clear();token=0;img=0;imgname[0]='\0';}
} CvMouseInfo;
void on_mouse(int evt, int x, int y, int flags, void* param);

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  i | input   | im%%04d.jpg | path to list of images or video }"
          "{  o | output  | train.yml   | path to annotation file in yml }"
          "{  s | start   | 1           | start index of input file format }"
          "{  h | help    | false       | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const char * path = parser.get<string>("input").c_str();
  const char * output_filename = parser.get<string>("output").c_str();
  const int start_index = parser.get<int>("start");
  const int display_help = parser.get<bool>("help");
  if (display_help){parser.printParams();return 0;}
  char filepath[1<<10]; memset(filepath,0,sizeof(filepath));
  cvNamedWindow("Annotator");
  CvMouseInfo mouse_info;
  cvSetMouseCallback("Annotator", on_mouse, &mouse_info);
  CvFileStorage * fs = cvOpenFileStorage(output_filename,0,CV_STORAGE_WRITE);
  CvSeqWriter writer; CvSeqReader reader;
  cvStartWriteStruct(fs,"frames",CV_NODE_SEQ,0,cvAttrList(0,0));
  cvStartWriteSeq(0,sizeof(CvSeq),sizeof(CvMouseInfo),cvCreateMemStorage(),&writer);

  for (int idx=start_index;; idx++){
    sprintf(filepath,path,idx);
    IplImage * img = cvLoadImage(filepath,1);
    if (!img){LOGW("File %s not found.",filepath);break;}else{LOGI("File %s loaded.",filepath);}
    mouse_info.img = img;
    strcpy(mouse_info.imgname,filepath);

    cvShowImage("Annotator",img);
 
    int key = cvWaitKey(-1)&0xff;
    if (key=='q' || key==27){break;}
    else if (key==' ' || key==32){}else{
      LOGW("unknown keyboard event: %c (%d,0x%x)\n",key,key,key);
    }

    if (mouse_info.centers.size()>0){
      cvStartWriteStruct(fs,0,CV_NODE_SEQ,0,cvAttrList(0,0));
      cvWriteString(fs,0,mouse_info.imgname);
      fprintf(stderr,"%s: ",filepath);
      for (int cc=0;cc<mouse_info.centers.size();cc++){
        cvWriteInt(fs,0,mouse_info.centers[cc].x);
        cvWriteInt(fs,0,mouse_info.centers[cc].y);
        fprintf(stderr," (%d,%d)",
                mouse_info.centers[cc].x,mouse_info.centers[cc].y);
      }
      fprintf(stderr,"\n");
      cvEndWriteStruct(fs);
      cvFlushSeqWriter(&writer);
    }
    mouse_info.clear();
  }

  cvEndWriteStruct(fs);
  cvDestroyAllWindows();
  cvReleaseFileStorage(&fs);

  return 0;
}

void on_mouse(int evt, int x, int y, int flags, void* param)
{
  static CvScalar colors[6] = {CV_RED, CV_GREEN, CV_BLUE, CV_YELLOW, CV_CYAN, CV_PURPLE};
  // if (evt>0){fprintf(stderr,"event: 0x%x\n",evt);}
  if (CV_EVENT_LBUTTONDOWN==evt){  // point
    int size = ((CvMouseInfo*)param)->centers.size();
    fprintf(stderr, "mouse event[%d]: (%d,%d)\n", size, x, y);
    ((CvMouseInfo*)param)->centers.push_back(cvPoint(x,y));
    ((CvMouseInfo*)param)->token = 1;
  }else if (CV_EVENT_RBUTTONDOWN==evt){  // cancel
    int size = ((CvMouseInfo*)param)->centers.size();
    if (size>0){
      fprintf(stderr, "mouse event[%d]: (%d,%d) removed\n", size-1, 
              ((CvMouseInfo*)param)->centers[size-1].x, 
              ((CvMouseInfo*)param)->centers[size-1].y);
      ((CvMouseInfo*)param)->centers.erase(size-1);
    }
  }
  if (((CvMouseInfo*)param)->img){
    IplImage * disp = cvCloneImage(((CvMouseInfo*)param)->img);
    for (int ii=0;ii<((CvMouseInfo*)param)->centers.size();ii++){
      cvCircle(disp,cvPoint(((CvMouseInfo*)param)->centers[ii].x,
                            ((CvMouseInfo*)param)->centers[ii].y),2,colors[ii%6],-1);
    }
    cvShowImage("Annotator",disp);
    cvReleaseImage(&disp);
  }
}

