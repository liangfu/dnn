dnn: A light-weight yet efficient framework for deep learning
=============================================================

.. image:: https://travis-ci.org/liangfu/dnn.svg?branch=master
  :target: https://travis-ci.org/liangfu/dnn

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
  :alt: LICENSE

Overview
--------

The Deep Neural Nets (DNN) library is a deep learning framework designed to be small in size, 
computationally efficient and portable.

We started the project as a fork of the popular `OpenCV <http://opencv.org/>`_ library,
while removing some components that is not tightly related to the deep learning framework.
Comparing to Caffe and many other implements, DNN is relatively independent to third-party libraries, 
(Yes, we don't require Boost and Database systems to be install before crafting your own network models)
and it can be more easily portable to mobile systems, like iOS, Android and RaspberryPi etc.

Available Modules
-----------------

The following features have been implemented:

 - Mini-batch based learning, with OpenMP support
 - YAML based network definition
 - Gradient checking for all implemented layers

The following modules are implemented in current version:

===================== ========================================================================================
  Module Name          Description																																					
===================== ========================================================================================
 `InputDataLayer`      Data Container Layer, for storing original input images															
--------------------- ----------------------------------------------------------------------------------------
 `ConvolutionLayer`    Convolutional Neural Network Layer, performs 2d convolution upon images							
--------------------- ----------------------------------------------------------------------------------------
 `SubSamplingLayer`    Sub-Sampling Layer, performs max-pooling operation																		 
--------------------- ----------------------------------------------------------------------------------------
 `FullConnectLayer`    Fully Connected Layer, with activation options, e.g. tanh, sigmoid, softmax, relu etc. 
--------------------- ----------------------------------------------------------------------------------------
 `RecurrentNNLayer`    vallina Recurrent Neural Network (RNN) Layer, for processing sequence data						  
--------------------- ----------------------------------------------------------------------------------------
 `CombineLayer`        Combine Layer, for combining output results from multiple different layers						  
===================== ========================================================================================

More modules will be available online !

Network Definition
~~~~~~~~~~~~~~~~~~

=============  ====================================================================================
Layer Type     Attributes
=============  ====================================================================================
`InputData`    `name`, `n_input_planes`, `input_height`, `input_width`, `seq_length`
`Convolution`  `name`, `visualize`, `n_output_planes`, `ksize`
`SubSampling`  `name`, `visualize`, `ksize`
`FullConnect`  `name`, `input_layer(optional)`, `visualize`, `n_output_planes`, `activation_type`
`RecurrentNN`  `name`, `n_output_planes`, `seq_length`, `time_index`, `activation_type`
`Combine`      `name`, `input_layers`, `visualize`, `n_output_planes`
=============  ====================================================================================

With the above parameters given in YAML format, one can simply define a network. 
For instance, a modifed lenet can be:

.. code-block:: yaml

 %YAML:1.0
 layers:
  - {type: InputData, name: input1, n_input_planes: 1, input_height: 28, input_width: 28, seq_length: 1}
  - {type: Convolution, name: conv1, visualize: 0, n_output_planes: 6, ksize: 5, stride: 1}
  - {type: SubSampling, name: pool1, visualize: 0, ksize: 2, stride: 2}
  - {type: Convolution, name: conv2, visualize: 0, n_output_planes: 16, ksize: 5, stride: 1}
  - {type: SubSampling, name: pool2, visualize: 0, ksize: 2, stride: 2}
  - {type: FullConnect, name: fc1, visualize: 0, n_output_planes: 10, activation_type: tanh}

Then, by ruuning network training program:

.. code-block:: bash

 $ network train --solver data/mnist/lenet_solver.xml

one can start to train a simple network right away. And this is the way the source code 
and data models are tested in Travis-Ci. 
(See `.travis.yml <https://github.com/liangfu/dnn/blob/master/.travis.yml>`_ in the root directory)

Installation
------------

`CMake <https://cmake.org>`_ is required for successfully compiling the project. 

Under root directory of the project:

.. code-block:: bash

 $ cd $DNN_ROOT
 $ mkdir build
 $ cmake .. 
 $ make -j4

License
-------

MIT
