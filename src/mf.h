#ifndef _MF_H
#define _MF_H

#include "model.h"

class SgdReadFilter: public tbb::filter {
  mf::Blocks& blocks_;
  mf::Blocks& blocks_test_;
  MF& mf_;
  FILE* fr_;
  uint32 isize_;
  int iter_, index_;
  bool f_io_;
  char* buf_;
public:
  SgdReadFilter(mf::Blocks& blocks, FILE* fr, int iter, \
	      bool f_io, MF& mf, mf::Blocks& blocks_test)
    : tbb::filter(serial_in_order), blocks_(blocks),
      fr_(fr), iter_(iter), index_(0),
      f_io_(f_io), mf_(mf), blocks_test_(blocks_test) {buf_ = (char*)malloc(64000000);}
  ~SgdReadFilter() {
    free(buf_);
  }
  //Branch misprediction at 1 per iteration
  void* operator()(void*) {
    if(f_io_) {
      if(fread(&isize_, 1, sizeof(isize_), fr_)) {
	fread(buf_, 1, isize_, fr_);
	mf::Block* block = blocks_.add_block();
	block->ParseFromArray(buf_, isize_);
	return block;
      }
      else {
#ifdef DETAILS
    e = Time::now();
    int nn;
	printf("iter#1\t%f\ttRMSE=%f\n",std::chrono::duration<float>(e-s).count(), sqrt(mf_.calc_mse(blocks_test_, nn)*1.0/nn));
#endif
	f_io_ = false;
    mf_.seteta(++iter_);
      }
    }
    if(!f_io_) {
      if(index_ < blocks_.block_size()) {
	const mf::Block& bk = blocks_.block(index_++);
	return (void*)&bk;
      }
      else {
#ifdef DETAILS
    e = Time::now();
    int nn;
	printf("iter#%d\t%f\ttRMSE=%f\n", iter_, std::chrono::duration<float>(e-s).count(), sqrt(mf_.calc_mse(blocks_test_, nn)*1.0/nn));
	//printf("iter#%d\t%f\n", iter_, std::chrono::duration<float>(e-s).count());
#endif
    mf_.seteta(++iter_);
	if(iter_ <= mf_.iter_) {
	  index_ = 0;
	  return (void*)&(blocks_.block(index_++));
	}
	else
	  return NULL;
      }
    }
  }
};

class SgdFilter: public tbb::filter {
  MF& mf_;
public:
  SgdFilter(MF& model): tbb::filter(parallel), mf_(model) {}
  void* operator()(void* block) {
    float q[mf_.dim_]={0.0};
    mf::Block* bk = (mf::Block*)block;
    const float lameta = 1.0-mf_.eta_*mf_.lambda_;
    int vid, j, i;
    float error, rating;
    for(i=0; i<bk->user_size(); i++) {
        const mf::User& user = bk->user(i);
        const int uid = user.uid();
        const int size = user.record_size();
        for(j=0; j<size-mf_.prefetch_stride_; j++) {
#ifdef FETCH
		const mf::User_Record& rec_fetch = user.record(j+mf_.prefetch_stride_);
		const int vid_fetch = rec_fetch.vid();
		prefetch_range((char*)(mf_.phi_[vid_fetch]), mf_.dim_*sizeof(float));
#endif
            memset(q, 0.0, sizeof(float)*mf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            rating = rec.rating();
            error = float(rating)
                - cblas_sdot(mf_.dim_, mf_.theta_[uid], 1, mf_.phi_[vid], 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
            error = mf_.eta_*error;
            cblas_saxpy(mf_.dim_, error, mf_.theta_[uid], 1, q, 1);
            cblas_saxpy(mf_.dim_, lameta-1.0, mf_.theta_[uid], 1, mf_.theta_[uid], 1);
            cblas_saxpy(mf_.dim_, error, mf_.phi_[vid], 1, mf_.theta_[uid], 1);
            cblas_saxpy(mf_.dim_, lameta, mf_.phi_[vid], 1, q, 1);
            cblas_scopy(mf_.dim_, q, 1, mf_.phi_[vid], 1);
	    mf_.bu_[uid] = lameta*mf_.bu_[uid] + error;
	    mf_.bv_[vid] = lameta*mf_.bv_[vid] + error;
        }
	//prefetch_range((char*)(mf_.theta_[bk->user(i+1).uid()]), mf_.dim_*sizeof(float));
	for(; j<size; j++) {
            memset(q, 0.0, sizeof(float)*mf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            rating = rec.rating();
            error = float(rating)
                - cblas_sdot(mf_.dim_, mf_.theta_[uid], 1, mf_.phi_[vid], 1)
                - mf_.bu_[uid] - mf_.bv_[vid] - mf_.gb_;
            error = mf_.eta_*error;
            cblas_saxpy(mf_.dim_, error, mf_.theta_[uid], 1, q, 1);
            cblas_saxpy(mf_.dim_, lameta-1.0, mf_.theta_[uid], 1, mf_.theta_[uid], 1);
            cblas_saxpy(mf_.dim_, error, mf_.phi_[vid], 1, mf_.theta_[uid], 1);
            cblas_saxpy(mf_.dim_, lameta, mf_.phi_[vid], 1, q, 1);
            cblas_scopy(mf_.dim_, q, 1, mf_.phi_[vid], 1);
	    mf_.bu_[uid] = lameta*mf_.bu_[uid] + error;
	    mf_.bv_[vid] = lameta*mf_.bv_[vid] + error;
	}
    }
    return NULL;
  }
};

#endif