OS:=$(shell uname -s)
CC = g++

USR_INCLUDE = /usr/local/include
USR_LIB = /usr/local/lib

ifeq ($(OS),Darwin)
	CCFLAGS = -std=c++11 -O3
	LDFLAGS = -ltbb -lprotobuf -lblas -lm
	INCLUDE = /opt/local/include
	LIB = /opt/local/lib
endif
ifeq ($(OS),Linux)
	CCFLAGS = -std=c++11 -O3 -m64
	LDFLAGS = -ltbb -lprotobuf -lmkl_rt -lm
	INCLUDE = /opt/intel/mkl/include
	LIB = /opt/intel/mkl/lib/intel64
endif

SOURCE = getdata.cc blocks.pb.cc

all: $(SOURCE)
	$(CC) $(CCFLAGS) -I$(USR_INCLUDE) -L$(USR_LIB) -L$(LIB) $(SOURCE) -o getdata $(LDFLAGS) -DDETAILS -fopenmp

clean:
	rm -f *.o getdata
