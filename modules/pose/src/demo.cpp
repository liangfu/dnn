/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    main.cpp
 *  Author:  Hilton Bristow
 *  Created: Jun 27, 2012
 */

#include <stdio.h>
// #include <boost/filesystem.hpp>
#include "PartsBasedDetector.hpp"
#include "Candidate.hpp"
#include "FileStorageModel.hpp"
// #ifdef WITH_MATLABIO
// 	#include "MatlabIOModel.hpp"
// #endif
#include "Visualize.hpp"
#include "types.hpp"
#include "nms.hpp"
#include "Rect3.hpp"
#include "DistanceTransform.hpp"

using namespace cv;
using namespace std;

char * cvGetFileExtension(char * fullname)
{
  int i,retval;
  int len=strlen(fullname);
  for (i=len-1;i>0;i--){
	if (fullname[i]=='.'){retval=i;break;}
  }
  // strncpy(suffix,fullname+retval,len-retval);
  return fullname+retval;
}

int main(int argc, char** argv) {

	// check arguments
	if (argc != 3 && argc != 4) {
		printf("Usage: PartsBasedDetector model_file image_file [depth_file]\n");
		exit(-1);
	}

	// determine the type of model to read
	// boost::scoped_ptr<Model> model;
	// string ext = boost::filesystem::path(argv[1]).extension().string();
	Model * model = 0;// = new Model();
	char * ext = cvGetFileExtension(argv[1]);//boost::filesystem::path(argv[1]).extension().string();
	// if (ext.compare(".xml") == 0 || ext.compare(".yaml") == 0) {
	if (!strcmp(ext,".xml") || !strcmp(ext,".yaml")) {
      //model.reset(new FileStorageModel);
      model = new FileStorageModel;
	}
// #ifdef WITH_MATLABIO
// 	else if (ext.compare(".mat") == 0) {
// 		model.reset(new MatlabIOModel);
// 	}
// #endif
	else {
		printf("Unsupported model format: %s\n", ext);
		exit(-2);
	}
	bool ok = model->deserialize(argv[1]);
	if (!ok) {
		printf("Error deserializing file\n");
		exit(-3);
	}

	// create the PartsBasedDetector and distribute the model parameters
	PartsBasedDetector<float> pbd;
	pbd.distributeModel(*model);

    // report model information
#if 1
	fprintf(stderr,"nparts:%d\n",model->nparts());
	fprintf(stderr,"nmixtures:%d\n",model->nmixtures());
    fprintf(stderr,"thresh:%f\n",model->thresh());
	fprintf(stderr,"binsize:%d\n",model->binsize());
	fprintf(stderr,"nscales:%d\n",model->nscales());
	fprintf(stderr,"flen:%d\n",model->flen());
	fprintf(stderr,"norient:%d\n",model->norient());
	fprintf(stderr,"ncomponents:%d\n",model->ncomponents());
#endif    

	// load the image from file
	// Mat_<float> depth;
	Mat im = imread(argv[2]);
        if (im.empty()) {
            printf("Image not found or invalid image format\n");
            exit(-4);
        }
	// if (argc == 4) {
	// 	depth = imread(argv[3], IMREAD_ANYDEPTH);
	// 	// convert the depth image from mm to m
	// 	depth = depth / 1000.0f;
	// }

	// detect potential candidates in the image
	vector<Candidate> candidates;
	pbd.detect(im, candidates);
	
	// display the best candidates
	Visualize visualize(model->name());
	SearchSpacePruning<float> ssp;
    Mat canvas;
	if (candidates.size() > 0) {
	    Candidate::sort(candidates);
	    Candidate::nonMaximaSuppression(im, candidates, 0.3);
        fprintf(stderr,"Number of candidates: %ld\n", candidates.size());
        
        cvtColor(im, canvas, COLOR_RGB2BGR);
	    visualize.candidates(im, candidates, canvas, true);
	    visualize.skeleton(model, canvas, candidates, canvas);
        visualize.image(canvas);
	    waitKey();
	}else{fprintf(stderr,"Number of candidates: %ld\n", candidates.size());}
	return 0;
}
