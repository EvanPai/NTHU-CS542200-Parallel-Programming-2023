NVCCFLAGS  = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
NVCC     = nvcc 
XCOMPILERFLAGS = -Xcompiler

CC = gcc
CFLAGS = -O3 -march=native 
MPFLAGS = -fopenmp 

NVCXX = nvc++
NVCXXFLAGS = -O0 -std=c++14 -fast -acc -gpu=cc60 -Minfo=accel
ACCFLAGS = -O3 -std=c++14 -fast -acc -gpu=cc60 -Minfo=accel

LDFLAGS = -lm

TARGETS = KMeans KMeans_CUDA KMeans_OpenMP

.PHONY: all
all: $(TARGETS)

KMeans: KMeans.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

KMeans_OpenACC: KMeans_OpenACC.c
	$(NVCXX) $(ACCFLAGS) $(XCOMPILERFLAGS) $(LDFLAGS)  -o $@ $<

KMeans_OpenMP: KMeans_OpenMP.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(MPFLAGS) -o $@ $<

KMeans_CUDA: KMeans_CUDA.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

KMeans_hybrid: KMeans_hybrid.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $(XCOMPILERFLAGS) $(MPFLAGS) -o $@ $<

KMeans_ACC: KMeans_ACC.cpp
	$(NVCXX) $(NVCXXFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)