#include "model.h"

unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
std::chrono::time_point<Time> s,e;
std::default_random_engine generator(seed);
std::normal_distribution<float> gaussian(0.0f, 1.0f);

/* two chunks, read-only and frequent write, should be seperated.
   Align the address of write chunk to cache line */
void MF::init() {
  //alloc
  bu_ = (float*)malloc((nu_+nv_)*sizeof(float));
  bv_ = bu_ + nu_;
  theta_  = (float**)malloc((nu_+nv_) * sizeof(float*));
  posix_memalign((void**)&theta_[0], CACHE_LINE_SIZE, (nu_+nv_)*dim_*sizeof(float));
  for(int i=1; i<nu_; i++)
    theta_[i] = theta_[i-1] + dim_;
  phi_ = theta_ + nu_;
  phi_[0] = theta_[nu_-1] + dim_;
  for(int i=1; i<nv_; i++)
    phi_[i] = phi_[i-1] + dim_;
  //init
#pragma omp parallel for
  for(int i=0;i<nu_;i++) {
    for(int j=0;j<dim_;j++)
      theta_[i][j]=gaussian(generator)*1e-2;
  }
#pragma omp parallel for
  for(int i=0;i<nv_;i++) {
    for(int j=0;j<dim_;j++)
      phi_[i][j]=gaussian(generator)*1e-2;
  }
#pragma omp parallel for
  for(int i=0;i<nu_+nv_; i++) bu_[i] = gaussian(generator)*1e-2;
}

void MF::seteta(int round) {
    eta_ = (float)(eta0_ * 1.0/pow(round,gam_));
}


float MF::calc_mse(mf::Blocks& blocks, int& ndata) {
  std::mutex mlock;
  const int bsize = blocks.block_size();
  float sloss=0.0;
  ndata=0;
#pragma omp parallel for
  for(int i=0; i<bsize; i++)
    {
      float sl = 0.0;
      int nn=0;
      const mf::Block& bk = blocks.block(i);
      int usize = bk.user_size();
      for(int j=0; j<usize; j++) {
        const mf::User& user = bk.user(j);
        int uid = user.uid();
        int rsize = user.record_size();
        nn+=rsize;
        for(int k=0; k<rsize; k++) {
          const mf::User_Record& rec = user.record(k);
          int vid = rec.vid();
          float rating = rec.rating();
          float error = float(rating)-cblas_sdot(dim_, theta_[uid], 1, phi_[vid], 1)
              -bu_[uid]-bv_[vid] - gb_;
          sl+=error*error;
        }
      }
      mlock.lock();
      sloss+=sl;
      ndata+=nn;
      mlock.unlock();
   }
    return sloss;
}

int MF::plain_read(const char* data, mf::Blocks& blocks) {
  FILE* fr = fopen(test_data_, "rb");
  char* buf = (char*)malloc(64000000);
  uint32 isize;
  mf::Block* bk;
  int nn = 0;
  while(fread(&isize, 1, sizeof(isize), fr)) {
    fread(buf, 1, isize, fr);
    bk = blocks.add_block();
    bk->ParseFromArray(buf, isize);
    for(int i=0;i<bk->user_size();i++) {
        const mf::User& user = bk->user(i);
        nn+=user.record_size();
    }
  }
  free(buf);
  fclose(fr);
  return nn;
}

void MF::read_model() {
  FILE* fp = fopen(model_, "rb");
  fread(&nv_,1,sizeof(int),fp);
  fread(&nu_,1,sizeof(int),fp);
  fread(&dim_,1,sizeof(int),fp);
  //read lambda
  fread(&lambda_,1,sizeof(float),fp);
  //read gb
  //fread(&gb,1,sizeof(float),fp);
  //read v
  for(int i=0; i<nv_; i++)  fread(&bv_[i],1,sizeof(float),fp);
  for(int i=0; i<nv_; i++) {
	  for(int j=0; j<dim_; j++) {
		  fread(&phi_[i][j],1,sizeof(float),fp);
	  }
  }
  //read u
  for(int i=0; i<nu_; i++) fread(&bu_[i],1,sizeof(float),fp);
  for(int i=0; i<nu_; i++) {
	  for(int j=0; j<dim_; j++) fread(&theta_[i][j],1,sizeof(float),fp);
  }
  fclose(fp);
}
void MF::save_model(int round) {
    char file[256];
    sprintf(file, "%s_%d", result_, round);
    FILE* fp = fopen(file,"wb");
    fwrite(&nv_,1,sizeof(int),fp);
    fwrite(&nu_,1,sizeof(int),fp);
    fwrite(&dim_,1,sizeof(int),fp);
    //write lambda
    fwrite(&lambda_,1,sizeof(float),fp);
    //write gb
    //fwrite(&gb,1,sizeof(float),fp);
    //write v
    for(int i=0; i<nv_; i++)  fwrite(&bv_[i],1,sizeof(float),fp);
    for(int i=0; i<nv_; i++) {
       for(int j=0; j<dim_; j++) {
           fwrite(&phi_[i][j],1,sizeof(float),fp);
       }
    }
    //write u
    for(int i=0; i<nu_; i++) fwrite(&bu_[i],1,sizeof(float),fp);
    for(int i=0; i<nu_; i++) {
        for(int j=0; j<dim_; j++) fwrite(&theta_[i][j],1,sizeof(float),fp);
    }
    fclose(fp);
}
void DPMF::save_model(int round) {
    char file[256];
    sprintf(file, "%s_%d", result_, round);
    FILE* fp = fopen(file,"wb");
    fwrite(&nv_,1,sizeof(int),fp);
    fwrite(&nu_,1,sizeof(int),fp);
    fwrite(&dim_,1,sizeof(int),fp);
    //write lambda
    fwrite(&lambda_r_,1,sizeof(float),fp);
    fwrite(&lambda_ub_,1,sizeof(float),fp);
    fwrite(&lambda_vb_,1,sizeof(float),fp);
    for(int i=0;i<dim_;i++) fwrite(&lambda_u_[i],1,sizeof(float),fp);
    for(int i=0;i<dim_;i++) fwrite(&lambda_v_[i],1,sizeof(float),fp);
    //write gb
    //fwrite(&gb,1,sizeof(float),fp);
    //write v
    for(int i=0; i<nv_; i++)  fwrite(&bv_[i],1,sizeof(float),fp);
    for(int i=0; i<nv_; i++) {
       for(int j=0; j<dim_; j++) {
           fwrite(&phi_[i][j],1,sizeof(float),fp);
       }
    }
    //write u
    for(int i=0; i<nu_; i++) fwrite(&bu_[i],1,sizeof(float),fp);
    for(int i=0; i<nu_; i++) {
        for(int j=0; j<dim_; j++) fwrite(&theta_[i][j],1,sizeof(float),fp);
    }
    fclose(fp);
}

void DPMF::read_hyper() {
  FILE* fp = fopen(model_, "rb");
  fread(&nv_,1,sizeof(int),fp);
  fread(&nu_,1,sizeof(int),fp);
  fread(&dim_,1,sizeof(int),fp);
  //read lambda
  fread(&lambda_r_,1,sizeof(float),fp);
  fread(&lambda_ub_,1,sizeof(float),fp);
  fread(&lambda_vb_,1,sizeof(float),fp);
  for(int i=0;i<dim_;i++) fread(&lambda_u_[i],1,sizeof(float),fp);
  for(int i=0;i<dim_;i++) fread(&lambda_v_[i],1,sizeof(float),fp);
  //read gb
  //fread(&gb,1,sizeof(float),fp);
  fclose(fp);
}

void DPMF::read_model() {
  FILE* fp = fopen(model_, "rb");
  fread(&nv_,1,sizeof(int),fp);
  fread(&nu_,1,sizeof(int),fp);
  fread(&dim_,1,sizeof(int),fp);
  //read lambda
  fread(&lambda_r_,1,sizeof(float),fp);
  fread(&lambda_ub_,1,sizeof(float),fp);
  fread(&lambda_vb_,1,sizeof(float),fp);
  for(int i=0;i<dim_;i++) fread(&lambda_u_[i],1,sizeof(float),fp);
  for(int i=0;i<dim_;i++) fread(&lambda_v_[i],1,sizeof(float),fp);
  //read gb
  //fread(&gb,1,sizeof(float),fp);
  //read v
  for(int i=0; i<nv_; i++)  fread(&bv_[i],1,sizeof(float),fp);
  for(int i=0; i<nv_; i++) {
	  for(int j=0; j<dim_; j++) {
		  fread(&phi_[i][j],1,sizeof(float),fp);
	  }
  }
  //read u
  for(int i=0; i<nu_; i++) fread(&bu_[i],1,sizeof(float),fp);
  for(int i=0; i<nu_; i++) {
	  for(int j=0; j<dim_; j++) fread(&theta_[i][j],1,sizeof(float),fp);
  }
  fclose(fp);
}

void DPMF::init(mf::Blocks& blocks) {
  //alloc
  bu_ = (float*)malloc((2*(nu_+nv_)+2*dim_)*sizeof(float));
  bv_ = bu_ + nu_;
  ur_ = bv_ + nv_;
  vr_ = ur_ + nu_;
  lambda_u_ = vr_ + nv_;
  lambda_v_ = lambda_u_ + dim_;
  theta_  = (float**)malloc((nu_+nv_) * sizeof(float*));
  posix_memalign((void**)&theta_[0], CACHE_LINE_SIZE, (nu_+nv_)*dim_*sizeof(float));
  for(int i=1; i<nu_; i++)
    theta_[i] = theta_[i-1] + dim_;
  phi_ = theta_ + nu_;
  phi_[0] = theta_[nu_-1] + dim_;
  for(int i=1; i<nv_; i++)
    phi_[i] = phi_[i-1] + dim_;
  //init
#pragma omp parallel for
  for(int i=0;i<nu_;i++) {
    for(int j=0;j<dim_;j++)
      theta_[i][j]=gaussian(generator)*1e-2;
  }
#pragma omp parallel for
  for(int i=0;i<nv_;i++) {
    for(int j=0;j<dim_;j++)
      phi_[i][j]=gaussian(generator)*1e-2;
  }
#pragma omp parallel for
  for(int i=0;i<nu_+nv_; i++) bu_[i] = gaussian(generator)*1e-2;
  for(int i=0;i<2*dim_;i++) lambda_u_[i] = 1e2;

  //noise
  noise_ = (float*)malloc(sizeof(float)*noise_size_);
#pragma omp parallel for
  for(int i=0; i<noise_size_; i++) noise_[i] = gaussian(generator);
  //precompute weights according to training data
  precompute_weight(blocks);
  //bookkeeping
  gcount = 0;
  gcountu = new uint64 [nu_] ();
  gcountv = new std::atomic<uint64> [nv_];
  gmutex = new std::mutex [nv_];
  //differentially private
  if (tau_ <= 0)  tau_ = nv_;
  if (epsilon_ <= 0.0f) bound_ = 1.0f;
  else bound_ = epsilon_*1.0/(4.0*25.0*tau_);
  uniform_int_ = std::uniform_int_distribution<>(0, noise_size_-tau_*(dim_+1)-1);
  assert(noise_size_ - tau_*(dim_+1) > 10000);
}

void DPMF::precompute_weight(mf::Blocks& blocks) {
    FILE* fr = fopen(train_data_, "rb");
    uint32 nn;
    char* buf = (char*)malloc(64000000);
    int uid, vid;
    int uc[nu_] = {0};
    int vc[nv_] = {0};
    while(fread(&nn,1,sizeof(nn),fr)) {
        fread(buf, 1, nn, fr);
        mf::Block* block = blocks.add_block();
        block->ParseFromArray(buf,nn);
        for(int i=0;i<block->user_size();i++) {
            const mf::User& user = block->user(i);
            uid = user.uid();
            const int size = user.record_size();
            for(int j=0; j<size; j++) {
                const mf::User_Record& rec = user.record(j);
                vid = rec.vid();
                uc[uid] += 1;
                vc[vid] += 1;
                ntrain_++;
            }
        }
    }
    for(int i=0;i<nu_;i++) {ur_[i] = (float)ntrain_/uc[i];}
    for(int i=0;i<nv_;i++) {vr_[i] = (float)ntrain_/vc[i];}
    free(buf);
    fclose(fr);
}

void DPMF::finish_round(mf::Blocks& blocks, mf::Blocks& blocks_test, int round) {
    finish_noise();
    int ntr, nt;
    float mse = calc_mse(blocks, ntr);
    float tmse = calc_mse(blocks_test, nt);
    printf("round #%d\tRMSE=%f\ttRMSE=%f\t", round, sqrt(mse*1.0/ntr), sqrt(tmse*1.0/nt));
    sample_hyper(blocks, mse);
    seteta_cutoff(round+1);
    e = Time::now();
    printf("%f\n", std::chrono::duration<float>(e-s).count());
    if(round>=100 && round%20==0) save_model(round);
}

void DPMF::finish_noise() {
    const int gc = gcount.load();
    int rndind;
#pragma omp parallel for
    for(int i=0; i<nu_; i++) {
        rndind = uniform_int_(generator);
        int uc = gc - gcountu[i];
        gcountu[i] = 0;
        cblas_saxpy(dim_, sqrt(temp_*eta_*uc), noise_+rndind, 1, theta_[i], 1);
        bu_[i] += sqrt(temp_*eta_*uc) * noise_[rndind+dim_];
    }
#pragma omp parallel for
    for(int i=0;i<nv_;i++) {
        rndind = uniform_int_(generator);
        int vc = gc - gcountv[i].load();
        gcountv[i]=0;
        cblas_saxpy(dim_, sqrt(temp_*eta_*vc), noise_+rndind, 1, phi_[i], 1);
        bv_[i] += sqrt(temp_*eta_*vc) * noise_[rndind+dim_];
    }
    gcount=0;
}


void DPMF::sample_hyper(mf::Blocks& blocks, float mse) {
    gamma_posterior(lambda_r_, hyper_a_, hyper_b_, mse, ntrain_);
    gamma_posterior(lambda_ub_, hyper_a_, hyper_b_, normsqr(bu_, nu_), nu_);
    gamma_posterior(lambda_vb_, hyper_a_, hyper_b_, normsqr(bv_, nv_), nv_);

    float normu[dim_]={0.0}, normv[dim_]={0.0};
    normsqr_col(theta_, dim_, nu_, normu);
    normsqr_col(phi_, dim_, nv_, normv);
#pragma omp parallel for
    for(int i=0;i<dim_;i++) {
        gamma_posterior(lambda_u_[i], hyper_a_, hyper_b_, normu[i], nu_);
        gamma_posterior(lambda_v_[i], hyper_a_, hyper_b_, normv[i], nv_);
    }
}

void DPMF::seteta_cutoff(int round) {
    eta_ = std::max(mineta_, (float)(eta0_ * 1.0/pow(round,gam_)));
}


void AdaptRegMF::init1() {
  init();
  const int piece = 4;
  int nn = nu_/piece;

  bu_old_ = (float*)malloc((nu_+nv_)*sizeof(float));
  bv_old_ = bu_old_ + nu_;

  theta_old_  = (float**)malloc((nu_) * sizeof(float*));
  posix_memalign((void**)&theta_old_[0], CACHE_LINE_SIZE, (nu_)*dim_*sizeof(float));
  for(int i=1; i<nu_; i++)
    theta_old_[i] = theta_old_[i-1] + dim_;

  nn = nv_/piece;
  phi_old_  = (float**)malloc((nv_) * sizeof(float*));
  int k;
  for(k=0;k<piece-1;k++) {
    posix_memalign((void**)&phi_old_[k*nn], CACHE_LINE_SIZE, (nn)*dim_*sizeof(float));
    for(int i=1; i<nn; i++)
      phi_old_[k*nn+i] = phi_old_[k*nn+i-1] + dim_;
  }
  posix_memalign((void**)&phi_old_[k*nn], CACHE_LINE_SIZE, (nn+nv_%piece)*dim_*sizeof(float));
  for(int i=1; i<nn+nv_%piece; i++)
    phi_old_[k*nn+i] = phi_old_[k*nn+i-1] + dim_;
  //init
#pragma omp parallel for
  for(int i=0;i<nu_;i++) {
    for(int j=0;j<dim_;j++) {
      theta_old_[i][j] = theta_[i][j];
    }
  }
#pragma omp parallel for
  for(int i=0;i<nv_;i++) {
    for(int j=0;j<dim_;j++) {
      phi_old_[i][j] = phi_[i][j];
    }
  }
#pragma omp parallel for
  for (int i=0; i <nu_+nv_; i++) bu_old_[i] = bu_[i];
}


void AdaptRegMF::set_etareg(int round) {
    eta_reg_ = (float)(eta0_reg_ * 1.0/pow(round,gam_));
}

void AdaptRegMF::plain_read_valid(const char* valid, mf::Blocks& blocks_valid) {
  FILE* fr = fopen(valid, "rb");
  char* buf = (char*)malloc(64000000);
  uint32 isize,usize;
  mf::Block* bk;
  int ii=0;
  Record rr;
  while(fread(&isize, 1, sizeof(isize), fr)) {
    fread(buf, 1, isize, fr);
    bk = blocks_valid.add_block();
    bk->ParseFromArray(buf, isize);
    usize = bk->user_size();
    for(int j=0; j<usize; j++) {
        const mf::User& user = bk->user(j);
        const int uid = user.uid();
        const int size = user.record_size();
        for(int k=0; k<size; k++) {
            const mf::User_Record& rec = user.record(k);
            const int vid = rec.vid();
            rr.u_ = uid;
            rr.v_ = vid;
            rr.r_ = (float)rec.rating();
            recsv_.push_back(rr);
        }
    }
  }
  std::random_shuffle ( recsv_.begin(), recsv_.end() );
  fclose(fr);
  free(buf);
}