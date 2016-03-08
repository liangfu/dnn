all: train/digitStruct.xml test/digitStruct.xml

train.tar.gz:
	wget http://ufldl.stanford.edu/housenumbers/train.tar.gz

test.tar.gz:
	wget http://ufldl.stanford.edu/housenumbers/test.tar.gz

train: train.tar.gz
	tar zxvf train.tar.gz

test: test.tar.gz
	tar zxvf test.tar.gz

train/mat2xml.m: train
	cp mat2xml.m train

test/mat2xml.m: test
	cp mat2xml.m test

train/digitStruct.xml: train/mat2xml.m
	cd train && matlab -nodisplay -r 'mat2xml;quit;' && cd ..

test/digitStruct.xml: test/mat2xml.m
	cd test  && matlab -nodisplay -r 'mat2xml;quit;' && cd ..

