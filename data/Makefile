OS:=$(shell uname -s)
CC = g++

USR_INCLUDE = /home/ziqiliu/include
USR_LIB = /home/ziqiliu/lib

ifeq ($(OS),Darwin)
	CCFLAGS = -std=c++11 -O3
	LDFLAGS = -ltbb -lprotobuf -lblas -lm
	INCLUDE = /opt/local/include
	LIB = /opt/local/lib
endif
ifeq ($(OS),Linux)
	CCFLAGS = -std=c++11 -O3 -m64
	LDFLAGS = -ltbb -lprotobuf -lmkl_rt -lm
	INCLUDE = /home/ziqiliu/intel/mkl/include
	LIB = /home/ziqiliu/intel/mkl/lib/intel64
endif

SOURCE = getdata.cc blocks.pb.cc

all: $(SOURCE)
	$(CC) $(CCFLAGS) -I$(INCLUDE) -I$(USR_INCLUDE) -L$(LIB) -L$(USR_LIB) $(SOURCE) -o getdata $(LDFLAGS) -DDETAILS -fopenmp

clean:
	rm -f *.o mf
