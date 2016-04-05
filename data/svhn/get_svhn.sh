#!/usr/bin/env bash

# wget -c http://ufldl.stanford.edu/housenumbers/train.tar.gz
wget -c http://ufldl.stanford.edu/housenumbers/test.tar.gz

# tar zxvf train.tar.gz
tar zxvf test.tar.gz

# cp mat2xml.m train
cp mat2xml.m test

# cd train && matlab -nodesktop -r mat2xml && cd ..
cd test && matlab -nodesktop -r mat2xml && cd ..

