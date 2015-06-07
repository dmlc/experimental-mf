#ifndef _MODEL_H
#define _MODEL_H

#include "util.h"

class MF {
public:
  MF(char* train_data, char* test_data, char* result, char* model,    \
     int dim, int iter, float eta, float gam, float lambda, float gb, \
     int nu, int nv, int fly, int stride)
    : train_data_(train_data), test_data_(test_data),          \
        result_(result), model_(model), dim_(dim), iter_(iter),   \
        eta_(eta), eta0_(eta), gam_(gam), lambda_(lambda), gb_(gb),     \
        nu_(nu), nv_(nv), data_in_fly_(fly), prefetch_stride_(stride) {}
  ~MF() {
    free(theta_[0]), free(theta_), free(bu_);
  }
  void  init();
  float calc_mse(mf::Blocks& blocks, int& ndata);
  void read_model();
  void save_model(int round);
  void seteta(int round);
  int plain_read(const char* data, mf::Blocks& blocks);
  float **theta_, **phi_, *bu_, *bv_;
  const char * const train_data_, * const test_data_, * const result_, *const model_;
  float gb_;
  int dim_, iter_;
  float eta_, gam_;
  float lambda_, eta0_;
  int   nu_, nv_, data_in_fly_, prefetch_stride_;//48B
};

class DPMF : public MF {
public:
  DPMF(char* train_data, char* test_data, char* result, char* model,  \
       int dim, int iter, float eta, float gam, float lambda,         \
       float gb, int nu, int nv, int fly, int stride, float hypera,   \
      float hyperb, float epsilon, int tau, int noise_size,           \
       float temp, float mineta)
      : MF(train_data, test_data, result, model, dim, iter,           \
           eta, gam, lambda, gb, nu, nv, fly, stride),                   \
      hyper_a_(hypera), hyper_b_(hyperb),                             \
      lambda_r_(1e0), lambda_ub_(1e2), lambda_vb_(1e2),               \
      epsilon_(epsilon), tau_(tau), noise_size_(noise_size),          \
        temp_(temp), mineta_(mineta), ntrain_(0), ntest_(0) {}
  ~DPMF() {
    free(theta_[0]), free(theta_), free(bu_), free(noise_);
    delete [] gcountu; delete [] gcountv; delete [] gmutex;}
  void init(mf::Blocks& blocks);
  void precompute_weight(mf::Blocks& blocks);
  void seteta_cutoff(int round);
  void read_model();
  void read_hyper();
  void save_model(int round);
  void finish_noise();
  void finish_round(mf::Blocks& blocks, mf::Blocks& blocks_test, int round);
  void sample_hyper(mf::Blocks& blocks, float mse);
  uint64 *gcountu;
  std::atomic<uint64> *gcountv;//128
  std::mutex *gmutex;
  std::uniform_int_distribution<> uniform_int_;
  float *ur_, *vr_, *noise_, *lambda_u_, *lambda_v_;
  const float hyper_a_, hyper_b_;//192
  float epsilon_, bound_;
  float lambda_r_, lambda_ub_, lambda_vb_;
  float temp_, mineta_;
  int noise_size_, tau_;
  int ntrain_, ntest_;
  char pad[64];
  std::atomic<uint64> gcount;
};

class AdaptRegMF : public MF {
public:
AdaptRegMF(char* train_data, char* test_data, char* valid_data, char* result, char* model, \
     int dim, int iter, float eta, float gam, float lambda, float gb, \
               int nu, int nv, int fly, int stride, int loss, int measure,\
        float eta_reg)
     : MF(train_data, test_data, result, model, dim, iter,\
          eta, gam, lambda, gb, nu, nv, fly, stride), valid_data_(valid_data),
        lam_u_(lambda), lam_v_(lambda), lam_b_(lambda), \
        loss_(loss), measure_(measure), eta_reg_(eta_reg), eta0_reg_(eta_reg) {}
    ~AdaptRegMF() {}
    void init1();
    inline void updateReg(int uid, int vid, float rating, float* p) {
        float pred = active( cblas_sdot(dim_, theta_[uid], 1, phi_[vid], 1) + bu_[uid] + bv_[vid] + gb_, loss_);
        float grad = cal_grad(rating, pred, loss_);
        updateUV(grad, uid, vid);
        updateBias(grad, uid, vid);
    }
    inline void updateUV(float grad, int uid, int vid) {
        float inner = cblas_sdot(dim_, theta_old_[uid], 1, phi_[vid], 1);
        lam_u_ = std::max(0.0f, lam_u_ - eta_reg_*eta_*grad*inner);
        inner = cblas_sdot(dim_, theta_[uid], 1, phi_old_[vid], 1);
        lam_v_ = std::max(0.0f, lam_v_ - eta_reg_*eta_*grad*inner);
    }

    inline void updateBias(float grad, int uid, int vid) {
        lam_b_ = std::max(0.0f, lam_b_ - eta_reg_*eta_*grad*(bu_old_[uid]+bv_old_[vid]));
    }
    void set_etareg(int round);
    void plain_read_valid(const char* valid, mf::Blocks& blocks_valid);
    std::vector<Record> recsv_;
    float **theta_old_, **phi_old_, *bu_old_, *bv_old_;
    const char* valid_data_;
    float eta_reg_, eta0_reg_;
    int loss_, measure_;
    char pad1[64];
    float lam_u_;
    char pad2[64];
    float lam_v_;
    char pad3[64];
    float lam_b_;
};

#endif