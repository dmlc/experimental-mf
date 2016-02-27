alg = 'mf'
nu = 480189
nv = 17770
traindata='~/works/data/netflix_protobuf_train_4by500'
testdata='~/works/data/netflix_protobuf_valid'
#nu = 1000990
#nv = 624961
#traindata='~/works/data/yahoo_protobuf_train_4by500'
#testdata='~/works/data/yahoo_protobuf_valid'

it=100
fly=24
dim=2048

#mf
eta=2.4e-2
lam=4e-2

#dpmf
eps=0.0
tau=0
mineta=2e-13

#admf
eta_reg=2e-2
#validdata='~/works/data/netflix_protobuf_valid'
validdata='~/works/data/yahoo_protobuf_valid'

import os
import sys

for eta in [4e-2]:
  for eta_reg in [5e-1]:
    for temp in [1e-1]:
      for gam in [1.0]:
        for dim in [128]:
          print './mf --alg %s --train %s --test %s --valid %s --nu %d --nv %d --eta %e --lambda %e --gam %f --result %s --iter %d --dim %d --fly %d --epsilon %f --tau %d --temp %e --mineta %e --eta_reg %e'%(alg,traindata,testdata,validdata,nu,nv,eta,lam,gam,'%s_dim%d'%(alg,dim),it,dim,fly,eps,tau,temp,mineta,eta_reg)
          sys.stdout.flush()
          os.system('./mf --alg %s --train %s --test %s --valid %s --nu %d --nv %d --eta %e --lambda %e --gam %f --result %s --iter %d --dim %d --fly %d --epsilon %f --tau %d --temp %e --mineta %e --eta_reg %e'%(alg,traindata,testdata,validdata,nu,nv,eta,lam,gam,'%s_dim%d'%(alg,dim),it,dim,fly,eps,tau,temp,mineta,eta_reg))
          sys.stdout.flush()

