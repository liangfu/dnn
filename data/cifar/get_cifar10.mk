#!/usr/bin/env bash

cifar-10-binary.tar.gz:
	wget -c https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

cifar-10-batches-bin: cifar-10-binary.tar.gz
	tar zxvf $<



