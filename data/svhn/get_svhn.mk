

all: train test
.PHONY: all

train.tar.gz:
	wget -c http://ufldl.stanford.edu/housenumbers/train.tar.gz

test.tar.gz:
	wget -c http://ufldl.stanford.edu/housenumbers/test.tar.gz

train: train.tar.gz
	tar zxvf train.tar.gz
	cp mat2xml.m train
	cd train && matlab -nodesktop -r mat2xml && cd ..

test: test.tar.gz
	tar zxvf test.tar.gz
	cp mat2xml.m test
	cd test && matlab -nodesktop -r mat2xml && cd ..

