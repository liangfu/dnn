
all: train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte

.PHONY: all

train-images-idx3-ubyte.gz: 
	wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz

train-labels-idx1-ubyte.gz:
	wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte.gz:
	wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte.gz:
	wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

train-images-idx3-ubyte: train-images-idx3-ubyte.gz
	gunzip train-images-idx3-ubyte.gz

train-labels-idx1-ubyte: train-labels-idx1-ubyte.gz
	gunzip train-labels-idx1-ubyte.gz

t10k-images-idx3-ubyte: t10k-images-idx3-ubyte.gz
	gunzip t10k-images-idx3-ubyte.gz

t10k-labels-idx1-ubyte: t10k-labels-idx1-ubyte.gz
	gunzip t10k-labels-idx1-ubyte.gz

