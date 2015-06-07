alg = 'admf'
traindata='../data/netflix_train_10blocks'
testdata='../data/netflix_test_1blocks'
validdata='../data/netflix_test_1blocks'
it=15
nu = 480189
nv = 17770
fly=8
mineta=2e-13
lam=5e-3
eps=0.0
eta_reg=2e-3

import os
import sys
#import numpy

for eta in [2e-2]:
  for temp in [1e-1]:
    for gam in [1.0]:
      for dim in [16]:
        print './mf --alg %s --mineta %e --epsilon %f --fly %d --gam %f --train %s --test %s --valid %s --eta_reg %e --result %s --eta %e --lambda %e --iter %d --dim %d --nu %d --nv %d'%(alg,mineta,eps,fly,gam,traindata,testdata,validdata,eta_reg,'%s_dim%d'%(alg,dim),eta,lam,it,dim,nu,nv)
        sys.stdout.flush()
        os.system('./mf --alg %s --mineta %e --epsilon %f --fly %d --gam %f --train %s --test %s --valid %s --eta_reg %e --result %s --eta %e --lambda %e --iter %d --dim %d --nu %d --nv %d'%(alg,mineta,eps,fly,gam,traindata,testdata,validdata,eta_reg,'%s_dim%d'%(alg,dim),eta,lam,it,dim,nu,nv))
        sys.stdout.flush()

