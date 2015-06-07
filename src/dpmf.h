#ifndef _DPMF_H
#define _DPMF_H

#include "model.h"

class SgldReadFilter: public tbb::filter {
public:
  mf::Blocks& blocks_;
  DPMF& dpmf_;
  int index_;
public:
  SgldReadFilter(mf::Blocks& blocks, DPMF& dpmf)
    : tbb::filter(serial_in_order), blocks_(blocks), index_(0), dpmf_(dpmf) {}
  ~SgldReadFilter() {
  }
  //Branch misprediction at 1 per iteration
  void* operator()(void*) {
      if(index_ < blocks_.block_size()) {
          const mf::Block& bk = blocks_.block(index_++);
          return (void*)&bk;
      } else return NULL;
    }
};


class SgldFilter: public tbb::filter {
  DPMF& dpmf_;
public:
  SgldFilter(DPMF& dpmf): tbb::filter(parallel), dpmf_(dpmf) {}
  void* operator()(void* block) {
    float q[dpmf_.dim_]={0.0};
    float p[dpmf_.dim_];
    mf::Block* bk = (mf::Block*)block;
    const float eta = dpmf_.eta_;
    const float scal = eta*dpmf_.ntrain_*dpmf_.bound_*dpmf_.lambda_r_;
    int vid, rating, j, i, gc, vc, uc;
    float error;
    for(i=0; i<bk->user_size(); i++) {
        const mf::User& user = bk->user(i);
        const int uid = user.uid();
        const int size = user.record_size();
        int thetaind = dpmf_.uniform_int_(generator);
        int phiind = dpmf_.uniform_int_(generator);
        for(j=0; j<size; j++) {
            memset(q, 0.0, sizeof(float)*dpmf_.dim_);
            const mf::User_Record& rec = user.record(j);
            vid = rec.vid();
            rating = rec.rating();

            dpmf_.gmutex[vid].lock();
            gc = dpmf_.gcount.fetch_add(1);
            vc = gc - dpmf_.gcountv[vid].exchange(gc);
            dpmf_.gmutex[vid].unlock();
            uc = gc - dpmf_.gcountu[uid];
            dpmf_.gcountu[uid] = gc;
            cblas_saxpy(dpmf_.dim_, sqrt(dpmf_.temp_*eta*uc), dpmf_.noise_+thetaind, 1, dpmf_.theta_[uid], 1);
            cblas_saxpy(dpmf_.dim_, sqrt(dpmf_.temp_*eta*vc), dpmf_.noise_+phiind, 1, dpmf_.phi_[vid], 1);
            dpmf_.bu_[uid] += sqrt(dpmf_.temp_*eta*uc) * dpmf_.noise_[thetaind+dpmf_.dim_];
            dpmf_.bv_[vid] += sqrt(dpmf_.temp_*eta*vc) * dpmf_.noise_[phiind+dpmf_.dim_];

            error = float(rating)
                - cblas_sdot(dpmf_.dim_, dpmf_.theta_[uid], 1, dpmf_.phi_[vid], 1)
                - dpmf_.bu_[uid] - dpmf_.bv_[vid] - dpmf_.gb_;
            error = scal*error;
            cblas_saxpy(dpmf_.dim_, error, dpmf_.theta_[uid], 1, q, 1);
            vsMul(dpmf_.dim_, dpmf_.lambda_u_, dpmf_.theta_[uid], p);
            cblas_saxpy(dpmf_.dim_, -eta*dpmf_.ur_[uid]*dpmf_.bound_, p, 1, dpmf_.theta_[uid], 1);
            cblas_saxpy(dpmf_.dim_, error, dpmf_.phi_[vid], 1, dpmf_.theta_[uid], 1);
            vsMul(dpmf_.dim_, dpmf_.lambda_v_, dpmf_.phi_[vid], p);
            cblas_saxpy(dpmf_.dim_, -eta*dpmf_.vr_[vid]*dpmf_.bound_, p, 1, dpmf_.phi_[vid], 1);
            cblas_saxpy(dpmf_.dim_, 1.0, q, 1, dpmf_.phi_[vid], 1);

            dpmf_.bu_[uid] = (1.0-eta*dpmf_.lambda_ub_*dpmf_.ur_[uid]*dpmf_.bound_)*dpmf_.bu_[uid] + error;
            dpmf_.bv_[vid] = (1.0-eta*dpmf_.lambda_vb_*dpmf_.vr_[vid]*dpmf_.bound_)*dpmf_.bv_[vid] + error;

            thetaind=thetaind+dpmf_.dim_+1; phiind=phiind+dpmf_.dim_+1;
        }
    }
    return NULL;
  }
};

#endif