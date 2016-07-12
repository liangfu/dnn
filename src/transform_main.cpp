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

#include "dnn.h"
#include "highgui.h"
#include "cxcore.h"
#include "cv.h"
#include "cvext.h"
#include "network.h"

typedef cv::CommandLineParser CvCommandLineParser;

int main(int argc, char * argv[])
{
  char keys[1<<12];
  sprintf(keys,
          "{  i | input   | im%%04d.jpg | path to list of images or video }"
          "{  o | output  | test.xml.gz | path to data file in xml }"
          "{  b | begin   | 1           | start index of input file format }"
          "{  s | scale   | 1           | ratio of scaled input image }"
          "{  t | move    | 0,0         | move image anchor to location }"
          "{ sz | size    | 64,64       | crop image to specified size }"
          "{  a | angle   | 0           | rotate to specified angle }"
          "{  v | visual  | 1           | visualize transformed image }"
          "{  n | noise   | 0           | add speckle noise image to input }"
          "{ eh | eqhist  | 1           | equalize image histogram }"
          "{ er | erode   | 0           | erode input image }"
          "{  h | help    | false       | display this help message    }");
  CvCommandLineParser parser(argc,argv,keys);
  const char * path = parser.get<string>("input").c_str();
  const char * output_filename = parser.get<string>("output").c_str();
  const int start_index = parser.get<int>("begin");
  const int display_help = parser.get<bool>("help");
  const int visualize = parser.get<int>("visual");
  const float scale = parser.get<float>("scale");
  const float theta = parser.get<float>("angle")/180.f*CV_PI;
  char move[1024]={0,}; strcpy(move,parser.get<string>("move").c_str());
  const float tx = atof(strtok((char*)move,(char*)",")), ty = atof(strtok(0,(char*)","));
  char crop[1024]={0,}; strcpy(crop,parser.get<string>("size").c_str());
  const int nr = atoi(strtok((char*)crop,(char*)",")), nc = atoi(strtok(0,(char*)","));
  const int noise = parser.get<int>("noise");
  const int eh = parser.get<int>("eqhist");
  const int er = parser.get<int>("erode");
  if (display_help){parser.printParams();return 0;}
  char filepath[1<<10]; memset(filepath,0,sizeof(filepath));
  CvMat * out = cvCreateMat(nr,nc,CV_32F);
  CvMat * warp_p = cvCreateMat(2,3,CV_32F); cvZero(warp_p);
  CvRNG rng = cvRNG(-1);

  int count=0;
  for (int idx=start_index;; idx++){
    sprintf(filepath,path,idx);
    IplImage * img = cvLoadImage(filepath,0);
    if (!img){break;}else{count++;}
  }  

  CvMat * testing = cvCreateMat(count,nr*nc,CV_32F);
  count=0;
  for (int idx=start_index;; idx++){
    sprintf(filepath,path,idx);
    IplImage * img = cvLoadImage(filepath,0);
    if (!img){
      LOGW("File %s not found.",filepath);break;
    }else{fprintf(stderr,"info: File %s loaded.\n",filepath);}
    CvMat mat_hdr;
    CvMat * mat = cvCloneMat(cvGetMat(img,&mat_hdr));
    CvMat * mat2 = cvCreateMat(mat->rows,mat->cols,CV_32F);
    cvConvert(mat,mat2);
    warp_p->data.fl[0]=scale*cos(theta);
    warp_p->data.fl[1]=sin(theta);
    warp_p->data.fl[3]=-sin(theta);
    warp_p->data.fl[4]=scale*cos(theta);
    warp_p->data.fl[2]=tx;
    warp_p->data.fl[5]=ty;
    icvWarp(mat2,out,warp_p); // cvPrintf(stderr,"%.1f ",warp_p);

    // contrast enhancement
    float avg = cvAvg(out).val[0];
    float sdv = cvSdv(out);
    cvSubS(out,cvScalar(avg),out);
    cvScale(out,out,127.f*0.4f/sdv);
    cvAddS(out,cvScalar(30),out);

    // clear boundary
    CvMat out_submat_hdr;
    cvGetRows(out,&out_submat_hdr,0,20); cvSet(&out_submat_hdr,cvScalar(0));
    cvGetCols(out,&out_submat_hdr,0,10); cvSet(&out_submat_hdr,cvScalar(0));
    cvGetRows(out,&out_submat_hdr,63-20,63); cvSet(&out_submat_hdr,cvScalar(0));

    if (eh){
      cvMinS(out,255,out); cvMaxS(out,40,out);
      cvEqualizeHistEx(out);
      cvScale(out,out,2.2f);
    }

    if (er){
      cvScale(out,out,1.6f);
      cvMinS(out,255,out); cvMaxS(out,0,out);
      cvErodeEx(out,1);
    }
    
    // add speckle noise to target image
    if (noise){
      for (int iter=0;iter<100;iter++){ 
        int ridx = cvRandInt(&rng)%64, cidx = cvRandInt(&rng)%64;
        int val = cvRandInt(&rng)%255; cvmSet(out,ridx,cidx,val);
      }
    }

    cvMinS(out,255,out); cvMaxS(out,0,out);

    CvMat testing_submat_hdr,out_reshape_hdr;
    cvGetRow(testing,&testing_submat_hdr,count);
    cvReshape(out,&out_reshape_hdr,0,1);
    cvCopy(&out_reshape_hdr,&testing_submat_hdr);

    // visualize
    if (visualize){
      cvRectangle(out,cvPoint(10,20),cvPoint(54,44),CV_WHITE);
      CV_SHOW(out);
    }
    cvReleaseMat(&mat);cvReleaseMat(&mat2);
    count++;
  }
  cvSave(output_filename,testing);
  
  cvReleaseMat(&testing);
  cvReleaseMat(&warp_p);
  cvReleaseMat(&out);
  cvDestroyAllWindows();

  return 0;
}

