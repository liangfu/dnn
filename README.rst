=====================
 The CompVis Library
=====================

Introduction
============

The CompVis library is built to demostrate its efficient and 
accurate implementation on a few Computer Vision tasks, including face
detection, face recognition, generic object tracking etc.

Available Modules
=================

The following modules are implemented in current version:

=======================  ========================================================================================
       module name       description
=======================  ========================================================================================
``image warping``        image resizing, rotating in an efficient way (with 3, 4 or 6 parameters)
``level sets``           image segmentation with composed energy function
``optical flow``         implements the inverse compositional algorithms
``AAM``                  active appearance model for facial image alignment
``LDA``                  fisher discriminant analysis (linear classifier)
``KFD``                  kernel fisher discriminant (nonlinear,guassian kernel)
``PWP``                  pixel-wise posterior, a level sets based tracking framework
``particle filter``      a probabilistic tracking framework
``cascade detector``     implements the classic viola-jones detection framework, with pre-trained feature sets
``sparse coding``        implements ``orthogonal matching pursuit (OMP)`` and ``basis pursuit (BP)`` algorithms
=======================  ========================================================================================

More modules will be available online !

Compilation
===========

Typically, ``qmake`` OR ``scons`` is required for successfully compiling the project. However, there is no such
restriction for Visual C++ developers, click the solution file ``compvis.sln`` and you're ready to go.

For advance usage, you can create Makefiles with qmake by running

.. code-block:: bash

 $ ./configure

under root directory of the project, or build directly with scons

.. code-block:: bash

 $ scons -u

Then try anything you want. 

*Note that it is not necessary to install qmake if you don't want to build the demo application*.

License
=======

The source code is released under `MIT license <https://github.com/liangfu/compvis/blob/master/LICENSE>`_ . There is no actual restrictions in (re)-distributing the code in any form.
