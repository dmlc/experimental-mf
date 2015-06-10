alg = 'mf'
nu = 480189
nv = 17770
traindata='../data/netflix_train_10blocks'
testdata='../data/netflix_test_1blocks'

it=15
fly=8
lam=5e-3

eps=0.0
tau=0
mineta=2e-13

eta_reg=2e-3
validdata='../data/netflix_test_1blocks'

import os
import sys
#import numpy

for eta in [2e-2]:
  for temp in [1e-1]:
    for gam in [1.0]:
      for dim in [16]:
        print './mf --alg %s --train %s --test %s --valid %s --nu %d --nv %d --eta %e --lambda %e --gam %f --result %s --iter %d --dim %d --fly %d --epsilon %f --tau %d --temp %e --mineta %e --eta_reg %e'%(alg,traindata,testdata,validdata,nu,nv,eta,lam,gam,'%s_dim%d'%(alg,dim),it,dim,fly,eps,tau,temp,mineta,eta_reg)
        sys.stdout.flush()
        os.system('./mf --alg %s --train %s --test %s --valid %s --nu %d --nv %d --eta %e --lambda %e --gam %f --result %s --iter %d --dim %d --fly %d --epsilon %f --tau %d --temp %e --mineta %e --eta_reg %e'%(alg,traindata,testdata,validdata,nu,nv,eta,lam,gam,'%s_dim%d'%(alg,dim),it,dim,fly,eps,tau,temp,mineta,eta_reg))
        sys.stdout.flush()

