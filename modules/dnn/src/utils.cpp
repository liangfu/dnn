/** -*- c++ -*- 
 *
 * \file   utils.cpp
 * \date   Sat May 21 17:47:52 2016
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
 * \brief  utility functions required in implement of dnn
 */
 
#include "_dnn.h"

double cvSdv(CvMat * src)
{
  CvScalar avg, sdv; 
  cvAvgSdv(src,&avg,&sdv); 
  return sdv.val[0];
}


