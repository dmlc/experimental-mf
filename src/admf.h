#ifndef _ADMF_H
#define _ADMF_H

#include "model.h"

class AdRegReadFilter: public tbb::filter {
  mf::Blocks& blocks_;
  mf::Blocks& blocks_test_;
  AdaptRegMF& admf_;
  FILE* fr_;
  uint32 isize_;
  int iter_, index_;
  bool f_io_;
  char* buf_;
public:
  AdRegReadFilter(mf::Blocks& blocks, FILE* fr, int iter,
    bool f_io, AdaptRegMF& admf, mf::Blocks& blocks_test)
    : tbb::filter(serial_in_order), blocks_(blocks), fr_(fr), iter_(iter), index_(0),
      f_io_(f_io), admf_(admf), blocks_test_(blocks_test)
  {buf_ = (char*)malloc(64000000);}
  ~AdRegReadFilter() {
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
	printf("iter#1\t%f\ttRMSE=%f\n",std::chrono::duration<float>(e-s).count(), sqrt(admf_.calc_mse(blocks_test_, nn)*1.0/nn));
#endif
	f_io_ = false;
    admf_.seteta(++iter_);
    admf_.set_etareg(iter_);
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
	printf("iter#%d\t%f\ttRMSE=%f\n", iter_, std::chrono::duration<float>(e-s).count(), sqrt(admf_.calc_mse(blocks_test_, nn)*1.0/nn));
	//printf("iter#%d\t%f\n", iter_, std::chrono::duration<float>(e-s).count());
#endif
    admf_.seteta(++iter_);
    admf_.set_etareg(iter_);
	if(iter_ <= admf_.iter_) {
	  index_ = 0;
	  return (void*)&(blocks_.block(index_++));
	}
	else
	  return NULL;
      }
    }
  }
};

class AdRegFilter: public tbb::filter {
  AdaptRegMF& admf_;
public:
  AdRegFilter(AdaptRegMF& model): tbb::filter(parallel), admf_(model) {}
  void* operator()(void* block) {
    float q[admf_.dim_]={0.0};
    mf::Block* bk = (mf::Block*)block;
    const float eta = admf_.eta_;
    int vid, j, i;
    float error, rating;
    for(i=0; i<bk->user_size(); i++) {
        const mf::User& user = bk->user(i);
        const int uid = user.uid();
        const int size = user.record_size();
        for(j=0; j<size; j++) {
            memset(q, 0.0, sizeof(float)*admf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            rating = rec.rating();
            cblas_scopy(admf_.dim_, admf_.theta_[uid], 1, admf_.theta_old_[uid], 1);
            cblas_scopy(admf_.dim_, admf_.phi_[vid], 1, admf_.phi_old_[vid], 1);
            float pred = active( cblas_sdot(admf_.dim_, admf_.theta_[uid], 1, admf_.phi_[vid], 1) + admf_.bu_[uid] + admf_.bv_[vid] + admf_.gb_, admf_.loss_);
            float error = cal_grad(rating, pred, admf_.loss_);
            error = eta*error;
            cblas_saxpy(admf_.dim_, error, admf_.theta_[uid], 1, q, 1);
            cblas_saxpy(admf_.dim_, -eta*admf_.lam_u_, admf_.theta_[uid], 1, admf_.theta_[uid], 1);
            cblas_saxpy(admf_.dim_, error, admf_.phi_[vid], 1, admf_.theta_[uid], 1);
            cblas_saxpy(admf_.dim_, 1.0f-eta*admf_.lam_v_, admf_.phi_[vid], 1, q, 1);
            cblas_scopy(admf_.dim_, q, 1, admf_.phi_[vid], 1);
            admf_.bu_old_[uid] = admf_.bu_[uid];
            admf_.bv_old_[vid] = admf_.bv_[vid];
            admf_.bu_[uid] = (1.0f-eta*admf_.lam_b_)*admf_.bu_[uid] + error;
            admf_.bv_[vid] = (1.0f-eta*admf_.lam_b_)*admf_.bv_[vid] + error;

            int ii = rand()%admf_.recsv_.size();
            admf_.updateReg(admf_.recsv_[ii].u_, admf_.recsv_[ii].v_, admf_.recsv_[ii].r_, q);
        }
    }
    return NULL;
  }
};

#endif