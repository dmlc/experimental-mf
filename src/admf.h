#ifndef _ADMF_H
#define _ADMF_H

#include "model.h"

class AdRegReadFilter: public tbb::filter {
  AdaptRegMF& admf_;
  mf::Blocks& blocks_test_;
  std::vector<std::vector<char> > pool_;
  FILE* fr_;
  uint32 isize_;
  int iter_, index_, pool_size_;
public:
  AdRegReadFilter(AdaptRegMF& admf, FILE* fr, mf::Blocks& blocks_test)
    : tbb::filter(serial_in_order), fr_(fr), iter_(1), index_(0),
      admf_(admf), blocks_test_(blocks_test) {
      pool_.resize(admf_.data_in_fly_*10);
      pool_size_ = pool_.size();
  }
  ~AdRegReadFilter() {}
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
          printf("iter#%d\t%f\ttRMSE=%f\n",iter_,std::chrono::duration<float>(e-s).count(), sqrt(admf_.calc_mse(blocks_test_, nn)*1.0/nn));
          //printf("iter#%d\t%f\n", iter_, std::chrono::duration<float>(e-s).count());
          if(iter_==admf_.iter_) return NULL;
          admf_.seteta(++iter_);
          admf_.set_etareg(iter_);
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

class AdRegFilter: public tbb::filter {
  AdaptRegMF& admf_;
public:
  AdRegFilter(AdaptRegMF& model): tbb::filter(parallel), admf_(model) {}
  void* operator()(void* block) {
    float q[admf_.dim_] __attribute__((aligned(CACHE_LINE_SIZE)));
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
            admf_.bu_[uid] = (1.0f-eta*admf_.lam_bu_)*admf_.bu_[uid] + error;
            admf_.bv_[vid] = (1.0f-eta*admf_.lam_bv_)*admf_.bv_[vid] + error;
        }
        int ii = rand()%admf_.recsv_.size();
        admf_.updateReg(admf_.recsv_[ii].u_, admf_.recsv_[ii].v_, admf_.recsv_[ii].r_);
    }
    return NULL;
  }
};

#endif