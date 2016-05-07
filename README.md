# Deep Neural Nets

[![Build Status](https://travis-ci.org/liangfu/dnn.svg?branch=master)](https://travis-ci.org/liangfu/dnn)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

## Introduction

The Deep Neural Nets (DNN) library is a deep learning framework designed to be small in size, computationally efficient and portable.

We started the project as a fork of the popular [OpenCV](http://opencv.org/) library,
while removing some components that is not tightly related to the deep learning framework.
Comparing to Caffe and many other implements, DNN is relatively independent to third-party libraries, 
(Yes, we don't require Boost and Database systems to be install before crafting your own network models)
and it can be more easily portable to mobile systems, like iOS, Android and RaspberryPi etc.

## Available Modules

The following features have been implemented:

 - Mini-batch based learning, with OpenMP support for parallel processing on CPU.
 - YAML based network definition
 - Gradient checking for all implemented layers

The following modules are implemented in current version:

* InputDataLayer       Data Container Layer, for storing original input images
* ConvolutionLayer     Convolutional Neural Network Layer, performs 2d convolution upon images
* SubSamplingLayer     Sub-Sampling Layer, performs max-pooling operation
* FullConnectLayer     Fully Connected Layer, with activation options, e.g. tanh, sigmoid, softmax, relu etc.
* RecurrentNNLayer     vallina Recurrent Neural Network (RNN) Layer, for processing sequence data
* CombineLayer         Combine Layer, for combining output results from multiple different layers

More modules will be available online !

## Compilation

[CMake](https://cmake.org) is required for successfully compiling the project. 

Under root directory of the project:

 $ cd $DNN_ROOT
 $ mkdir build
 $ cmake .. 
 $ make -j4

Then try anything you want. 

## License

MIT
