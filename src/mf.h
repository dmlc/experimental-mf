#ifndef _MF_H
#define _MF_H
#include <thread>

#include "model.h"
class SgdReadFilter: public tbb::filter {
  MF& mf_;
  const mf::Blocks& blocks_test_;
  std::vector<std::vector<char> > pool_;
  FILE* fr_;
  int pool_size_;
  int index_;
  uint32 isize_;
  int iter_;

public:
  SgdReadFilter(MF& mf, FILE* fr, mf::Blocks& blocks_test)
      : tbb::filter(serial_in_order), iter_(1), index_(0),
      mf_(mf), blocks_test_(blocks_test), fr_(fr) {
      pool_.resize(mf_.data_in_fly_*10);
      pool_size_ = pool_.size();
  }
  ~SgdReadFilter() {}
  void* operator()(void*) {
      if(fread(&isize_,1,sizeof(isize_),fr_)) {
          pool_[index_].resize(isize_);
          fread((char*)pool_[index_].data(),1,isize_,fr_);
          std::vector<char>& b = pool_[index_++];
          index_ %= pool_size_;
          return &b;
      }
      else {
          e = Time::now();
          int nn;
          printf("iter#%d\t%f\ttRMSE=%f\n",iter_,std::chrono::duration<float>(e-s).count(), sqrt(mf_.calc_mse(blocks_test_, nn)*1.0/nn));
          //printf("iter#%d\t%f\n", iter_, std::chrono::duration<float>(e-s).count());
          if(iter_==mf_.iter_) return NULL;
          mf_.seteta(++iter_);
          fseek(fr_,0,SEEK_SET);
          fread(&isize_,1,sizeof(isize_),fr_);
          pool_[index_].resize(isize_);
          fread((char*)pool_[index_].data(),1,isize_,fr_);
          std::vector<char>& b = pool_[index_++];
          index_ %= pool_size_;
          return &b;
      }
  }
};

class ParseFilter: public tbb::filter {
public:
    std::vector<mf::Block> pool_;
    std::mutex lo_;
    int pool_size_;
    char pad[CACHE_LINE_SIZE];
    int index_;
ParseFilter(int fly): tbb::filter(parallel), index_(0) {
        pool_.resize(fly*10);
        pool_size_ = pool_.size();
    }
    void* operator()(void* chunk) {
        std::vector<char>* p = (std::vector<char>*)chunk;
        lo_.lock();
        mf::Block& bk = pool_[index_++];
        index_ %= pool_size_;
        lo_.unlock();
        bk.ParseFromArray(p->data(), p->size());
        return &bk;
    }
};

class SgdFilter: public tbb::filter {
  MF& mf_;
public:
  SgdFilter(MF& model): tbb::filter(parallel), mf_(model) {}
  void* operator()(void* block) {
    float q[mf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
    const int pad = padding(mf_.dim_);
    mf::Block* bk = (mf::Block*)block;
    const float lameta = 1.0-mf_.eta_*mf_.lambda_;
    int vid, j, i;
    float error, rating, *theta, *phi;
    for(i=0; i<bk->user_size(); i++) {
        const mf::User& user = bk->user(i);
        const int uid = user.uid();
        theta = (float*)__builtin_assume_aligned(mf_.theta_[uid], CACHE_LINE_SIZE);
        const int size = user.record_size();
        for(j=0; j<size-mf_.prefetch_stride_; j++) {
#ifdef FETCH
            const mf::User_Record& rec_fetch = user.record(j+mf_.prefetch_stride_);
            const int vid_fetch = rec_fetch.vid();
            prefetch_range((char*)(mf_.phi_[vid_fetch]), pad*sizeof(float));
#endif
            memset(q, 0.0, sizeof(float)*mf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            phi = (float*)__builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
            rating = rec.rating();
            error = float(rating)
                - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
            error = mf_.eta_*error;
            cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
            cblas_saxpy(mf_.dim_, lameta-1.0, theta, 1, theta, 1);
            cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
            cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
            cblas_scopy(mf_.dim_, q, 1, phi, 1);
            mf_.bu_[uid] = lameta*mf_.bu_[uid] + error;
            mf_.bv_[vid] = lameta*mf_.bv_[vid] + error;
        }
        //prefetch_range((char*)(mf_.theta_[bk->user(i+1).uid()]), pad*sizeof(float));
        for(; j<size; j++) {
            memset(q, 0.0, sizeof(float)*mf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            phi = (float*)__builtin_assume_aligned(mf_.phi_[vid], CACHE_LINE_SIZE);
            rating = rec.rating();
            error = float(rating)
                - cblas_sdot(mf_.dim_, theta, 1, phi, 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
            error = mf_.eta_*error;
            cblas_saxpy(mf_.dim_, error, theta, 1, q, 1);
            cblas_saxpy(mf_.dim_, lameta-1.0, theta, 1, theta, 1);
            cblas_saxpy(mf_.dim_, error, phi, 1, theta, 1);
            cblas_saxpy(mf_.dim_, lameta, phi, 1, q, 1);
            cblas_scopy(mf_.dim_, q, 1, phi, 1);
            mf_.bu_[uid] = lameta*mf_.bu_[uid] + error;
            mf_.bv_[vid] = lameta*mf_.bv_[vid] + error;
        }
    }
    return NULL;
  }
};

#endif
