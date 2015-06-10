#include "model.h"
#include "mf.h"
#include "dpmf.h"
#include "admf.h"

void show_help() {
    printf("Usage:\n");
    printf("./mf\n");
    printf("--train      xxx       : xxx is the file name of the binary training data.\n");
    printf("--nu         int       : number of users.\n");
    printf("--nv         int       : number of items.\n");
    printf("--test       xxx       : xxx is the file name of the binary test data.\n");
    printf("--valid      [xxx]     : xxx is the file name of the binary validation data.\n");
    printf("--result     [xxx]     : save your model in name xxx.\n");
    printf("--model      [xxx]     : read your model in name xxx.\n");
    printf("--alg        [xxx]     : xxx can be {mf, dpmf, admf}.\n");
    printf("--dim        [int]     : low rank of the model.\n");
    printf("--iter       [int]     : number of iterations.\n");
    printf("--fly        [int]     : number of threads.\n");
    printf("--stride     [int]     : prefetch strides.\n");
    printf("--eta        [float]   : learning rate.\n");
    printf("--lambda     [float]   : regularizer.\n");
    printf("--gam        [float]   : decay of learning rate.\n");
    printf("--bias       [float]   : global bias (important for accuracy).\n");
    printf("--mineta     [float]   : minimum learning rate (sometimes used in SGLD).\n");
    printf("--epsilon    [float]   : sensitivity of differentially privacy.\n");
    printf("--tau        [int]     : maximum of ratings among all the users (usually after trimming your data).\n");
    printf("--temp       [float]   : temprature in SGLD (can accelarate the convergence).\n");
    printf("--noise_size [int]     : the Gaussian numbers lookup table.\n");
    printf("--eta_reg    [float]   : the learning rate for estimating regularization parameters.\n");
    printf("--loss       [int]     : the loss type can be {least square, 0-1 logistic regression}.\n");
    printf("--measure    [int]     : support RMSE.\n");
}

void run(MF& mf) {
  mf::Blocks blocks, blocks_test;
  mf.init();
  if(mf.model_ != NULL) mf.read_model();
  plain_read(mf.test_data_, blocks_test);
  FILE* fr = fopen(mf.train_data_, "rb");
  SgdReadFilter read_f(blocks, fr, 1, true, mf, blocks_test);
  SgdFilter sgd_f(mf);
  tbb::pipeline p;
  p.add_filter(read_f);
  p.add_filter(sgd_f);
  s = Time::now();
  p.run(mf.data_in_fly_);
  fclose(fr);
}
void run(DPMF& dpmf) {
  mf::Blocks blocks, blocks_test;
  dpmf.init(blocks);
  plain_read(dpmf.test_data_, blocks_test);
  if(dpmf.model_ != NULL) dpmf.read_hyper();
  SgldReadFilter read_f(blocks, dpmf);
  SgldFilter sgld_f(dpmf);
  tbb::pipeline p;
  p.add_filter(read_f);
  p.add_filter(sgld_f);
  s = Time::now();
  for(int i=1; i<=dpmf.iter_; i++) {
      p.run(dpmf.data_in_fly_);
      read_f.index_ = 0;
      dpmf.finish_round(blocks, blocks_test, i);
  }
}

void run(AdaptRegMF& admf) {
  mf::Blocks blocks, blocks_test, blocks_valid;
  admf.init1();
  plain_read(admf.test_data_, blocks_test);
  admf.plain_read_valid(admf.valid_data_, blocks_valid);
  FILE* fr = fopen(admf.train_data_, "rb");
  AdRegReadFilter read_f(blocks, fr, 1, true, admf, blocks_test);
  AdRegFilter admf_f(admf);
  tbb::pipeline p;
  p.add_filter(read_f);
  p.add_filter(admf_f);
  s = Time::now();
  p.run(admf.data_in_fly_);
  fclose(fr);
}

int main(int argc, char** argv) {
  char *train_data=NULL, *test_data=NULL, *result=NULL, *alg=NULL, *model=NULL;
  int dim = 128, iter = 15, tau=0, nu=0, nv=0, fly=8, stride=2;
  float eta = 2e-2, lambda = 5e-3, gam=1.0f, mineta=1e-13;
  float epsilon=0.0f, hypera=1.0f, hyperb=100.0f, temp=1.0f;
  float g_bias = 2.76f;
  int noise_size = 2000000000;
  int loss = 0;
  int measure = 0;
  float eta_reg = 2e-3f;
  char* valid_data=NULL;
  for(int i = 1; i < argc; i++) {
      if(!strcmp(argv[i], "--train"))            train_data = argv[++i];
      else if(!strcmp(argv[i], "--test"))        test_data = argv[++i];
      else if(!strcmp(argv[i], "--valid"))       valid_data = argv[++i];
      else if(!strcmp(argv[i], "--result"))      result = argv[++i];
      else if(!strcmp(argv[i], "--model"))       model = argv[++i];
      else if(!strcmp(argv[i], "--alg"))         alg = argv[++i];
      else if(!strcmp(argv[i], "--dim"))         dim  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--iter"))        iter = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--nu"))          nu  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--nv"))          nv  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--fly"))         fly  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--stride"))      stride  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--eta"))         eta = atof(argv[++i]);
      else if(!strcmp(argv[i], "--lambda"))      lambda = atof(argv[++i]);
      else if(!strcmp(argv[i], "--gam"))         gam = atof(argv[++i]);
      else if(!strcmp(argv[i], "--bias"))        g_bias = atof(argv[++i]);
      else if(!strcmp(argv[i], "--mineta"))      mineta = atof(argv[++i]);
      else if(!strcmp(argv[i], "--epsilon"))     epsilon = atof(argv[++i]);
      else if(!strcmp(argv[i], "--tau"))         tau = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--hypera"))      hypera = atof(argv[++i]);
      else if(!strcmp(argv[i], "--hyperb"))      hyperb = atof(argv[++i]);
      else if(!strcmp(argv[i], "--temp"))        temp = atof(argv[++i]);
      else if(!strcmp(argv[i], "--noise_size"))  noise_size  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--eta_reg"))     eta_reg = atof(argv[++i]);
      else if(!strcmp(argv[i], "--loss"))        loss  = atoi(argv[++i]);
      else if(!strcmp(argv[i], "--measure"))     measure  = atoi(argv[++i]);
      else {
	  printf("%s, unknown parameters, exit\n", argv[i]);
	  exit(1);
      }
  }
  if (train_data == NULL || nu == 0 || nv == 0) {
      printf("Note that train_data/#users/#items are not optional!\n");
      show_help();
      exit(1);
  }
  if(!strcmp(alg, "mf") || alg==NULL) {
      MF mf(train_data, test_data, result, model, dim, iter, eta, gam, lambda, \
            g_bias, nu, nv, fly, stride);
      run(mf);
  }
  else if (!strcmp(alg, "dpmf")) {
      DPMF dpmf(train_data, test_data, result, model, dim, iter, eta, gam, lambda, \
                g_bias, nu, nv, fly, stride, hypera, hyperb, epsilon, tau, \
                noise_size, temp, mineta);
      run(dpmf);
  }
  else if (!strcmp(alg, "admf")) {
      AdaptRegMF admf(train_data, test_data, valid_data, result, model, dim, iter, eta, gam, \
                      lambda, g_bias, nu, nv, fly, stride, loss, measure, eta_reg);
      run(admf);
  }
  else {
    printf("Pleae select a solver: mf/dpmf/admf\n");
    exit(2);
  }
 return 0;
}