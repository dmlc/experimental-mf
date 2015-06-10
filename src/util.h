#ifndef _UTIL_H
#define _UTIL_H

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <string>
#include <random>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <tbb/pipeline.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include "blocks.pb.h"
#ifdef __APPLE__
extern "C"
{
#include <cblas.h>
}
extern "C"
{
  void cblas_saxpy(const int N, const float alpha, const float *X,
		   const int incX, float *Y, const int incY);
  void cblas_scopy(const int N, const float *X, const int incX,
		   float *Y, const int incY);
  float cblas_sdot(const int N, const float  *X, const int incX,
		    const float  *Y, const int incY);
}
#else
#include "mkl.h"
#endif
typedef unsigned long long uint64;
typedef unsigned int       uint32;
typedef std::chrono::high_resolution_clock Time;
#ifdef LEVEL1_DCACHE_LINESIZE
#define CACHE_LINE_SIZE LEVEL1_DCACHE_LINESIZE
#else
#define CACHE_LINE_SIZE 64
#endif

typedef struct {
    int u_, v_;
    float r_;
} Record;

extern std::chrono::time_point<Time> s,e;
extern std::default_random_engine generator;
extern std::normal_distribution<float> gaussian;

#ifdef FETCH
inline void prefetch_range(char *addr, size_t len) {
  char *cp;
  char *end = addr + len;

  for (cp = addr; cp < end; cp += CACHE_LINE_SIZE)
    __builtin_prefetch(cp,1,0);
}
#endif

inline void align_alloc(float** u, int nu, int dim) {
    int piece = nu/1000000+1;
    int nn = nu/piece;
    int k;
    for(k=0; k<piece-1; k++) {
        posix_memalign((void**)&u[k*nn], CACHE_LINE_SIZE, nn*dim*sizeof(float));
        for(int i=1; i<nn; i++)
            u[k*nn+i] = u[k*nn+i-1] + dim;
    }
    posix_memalign((void**)&u[k*nn], CACHE_LINE_SIZE, (nn+nu%piece)*dim*sizeof(float));
    for(int i=1; i<nn+nu%piece; i++)
        u[k*nn+i] = u[k*nn+i-1] + dim;
}

inline void plain_read(const char* data, mf::Blocks& blocks) {
  FILE* fr = fopen(data, "rb");
  char* buf = (char*)malloc(64000000);
  uint32 isize;
  mf::Block* bk;
  while(fread(&isize, 1, sizeof(isize), fr)) {
    fread(buf, 1, isize, fr);
    bk = blocks.add_block();
    bk->ParseFromArray(buf, isize);
  }
  free(buf);
  fclose(fr);
}

inline float active(float val, int type) {
  switch(type) {
    case 0: return val;                     //least square
    case 1: return 1.0f/(1.0f+expf(-val));  //sigmoid
  }
}
inline float cal_grad(float r, float pred, int type) {
  switch(type) {
    case 0: return r - pred;          //least square
    case 1: return r - pred;          //0-1 logistic regression
  }
}

inline float next_float(){
    return static_cast<float>( rand() ) / (static_cast<float>( RAND_MAX )+1.0);
}

inline float next_float2(){
    return (static_cast<float>( rand() ) + 1.0 ) / (static_cast<float>(RAND_MAX) + 2.0);
}

inline float normsqr(float* x, int num) {
   return cblas_sdot(num, x, 1, x, 1);
}

inline float sample_normal(){
    float x,y,s;
    do{
        x = 2 * next_float2() - 1.0;
        y = 2 * next_float2() - 1.0;
        s = x*x + y*y;
    }while( s >= 1.0 || s == 0.0 );

    return x * sqrt( -2.0 * log(s) / s ) ;
}

inline float sample_gamma( float alpha, float beta ) {
    if ( alpha < 1.0 ) {
        float u;
        do {
            u = next_float();
        } while (u == 0.0);
        return sample_gamma(alpha + 1.0, beta) * pow(u, 1.0 / alpha);
    } else {
        float d,c,x,v,u;
        d = alpha - 1.0/3.0;
        c = 1.0 / sqrt( 9.0 * d );
        do {
            do {
                x = sample_normal();
                v = 1.0 + c*x;
            } while ( v <= 0.0 );
            v = v * v * v;
            u = next_float();
        } while ( (u >= (1.0 - 0.0331 * (x*x) * (x*x)))
            && (log(u) >= (0.5 * x * x + d * (1.0 - v + log(v)))) );
        return d * v / beta;
    }
}

inline void gamma_posterior( float &lambda, float prior_alpha, float prior_beta, float psum_sqr, float psum_cnt ){
    float alpha = prior_alpha + 0.5*psum_cnt;
    float beta  = prior_beta + 0.5*psum_sqr;
    lambda = sample_gamma( alpha, beta );
}

inline void normsqr_col(float** m, int d, int size, float* norm) {
#pragma omp parallel for
    for(int i=0; i<d; i++) {
        for(int j=0; j<size; j++) norm[i] += m[j][i]*m[j][i];
    }
}

#endif