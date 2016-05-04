================
Deep Neural Nets
================

.. image:: https://travis-ci.org/liangfu/dnn.svg  
 :target: https://travis-ci.org/liangfu/dnn

Introduction
============

The design principle of the Deep Neural Nets (DNN) library is to make it 
small in size, efficient computation and portable to multiple operating systems.
We started the project as a fork of the popular `OpenCV <http://opencv.org/>`_ library,
while removing some components that is not tightly related to the deep learning framework, 
e.g. motion history imaging, camera calibration, kalman filter, haar feature based object detection etc.

Available Modules
=================

The following features have been implemented:

 - mini-batch based learning
 - YAML based network definition
 - gradient checking for all implemented layers

The following modules are implemented in current version:

=======================  ========================================================================================
       module name       description
=======================  ========================================================================================
``InputDataLayer``       data container layer, for storing original input images
``ConvolutionLayer``     performs 2d convolution upon images
``SubSamplingLayer``     performs max-pooling operation
``FullConnectLayer``     full connection layer, with activation options, e.g. tanh, sigmoid, softmax, relu etc.
``RecurrentNNLayer``     vallina RNN layer
=======================  ========================================================================================

More modules will be available online !

Compilation
===========

`CMake <https://cmake.org>`_ is required for successfully compiling the project. 

Under root directory of the project:

.. code-block:: bash

 $ cd $DNN_ROOT
 $ mkdir build
 $ cmake .. 
 $ make -j4

Then try anything you want. 

License
=======

MIT
