import os
from random import shuffle
fr = open('hugewiki','r')

b=20
ratio=0.8
fr.readline()
title = fr.readline()
nu = 0
nv = 0
nr = 0
gntrain = 0
gntest = 0
maxntrain = 0
maxntest = 0
lntrain=[]
lntest=[]
maxr =20.746284
scal = 4.1492568
(nv,nu,nr) = map(int,title.split())
print '%d %d %d\n'%(nv, nu, nr)
for i in range(b):
    f=open('d%d'%i,'w')
    if i == b-1:
	tt = nr - i*(nr/b)
	f.write('%d\n'%(tt))
        for j in range(tt):
	    line = fr.readline()
	    l = line.split()
	    v = int(l[0])-1
	    u = int(l[1])-1
	    r = float(l[2])/scal
	    f.write('%d,%d,%f\n'%(u,v,r))
    else:
	f.write('%d\n'%(nr/b))
	for j in range(nr/b):
	    line = fr.readline()
	    l = line.split()
	    v = int(l[0])-1
	    u = int(l[1])-1
	    r = float(l[2])/scal
	    f.write('%d,%d,%f\n'%(u,v,r))
    f.close()
fr.close()
print 'split raw done\n'

for i in range(b):
    data = []
    f=open('d%d'%i,'r')
    fw=open('r%d'%i,'w')
    f.readline()
    lines = f.readlines()
    for line in lines:
        t = line.split(',')
	u = int(t[0])
	v = int(t[1])
	r = float(t[2])
	data.append((u,v,r))
    shuffle(data)
    shuffle(data)
    fw.write('%d\n'%len(data))
    for d in data:
        fw.write('%d,%d,%f\n'%(d[0],d[1],d[2]))
    f.close()
    fw.close()
        
print 'shuffle done\n'

for i in range(b):
    f=open('r%d'%i,'r')
    ftrain=open('train%d'%i,'w')
    ftest=open('test%d'%i, 'w')
    line=f.readline()
    n = int(line)
    ntrain = int(n*ratio)
    ntest = n-ntrain
    gntrain += ntrain
    gntest += ntest
    lntrain.append(ntrain)
    lntest.append(ntest)
    if ntrain > maxntrain:
        maxntrain = ntrain
    if ntest > maxntest:
        maxntest = ntest
    lines=f.readlines()
    ftrain.write('%d\n'%ntrain)
    for j in range(ntrain):
        t = (lines[j]).split(',')
	u = int(t[0])
	v = int(t[1])
	r = float(t[2])
	ftrain.write('%d,%d,%f\n'%(u,v,r))
    ftest.write('%d\n'%ntest)
    for j in range(ntrain,n):
        t = (lines[j]).split(',')
	u = int(t[0])
	v = int(t[1])
	r = float(t[2])
	ftest.write('%d,%d,%f\n'%(u,v,r))
    f.close()
    ftrain.close()
    ftest.close()
    
print 'split train and test done\n'
#graphchi train
fw=open('wiki_graphchi_train','w')
fw.write('%%MatrixMarket matrix coordinate real general\n%d %d %d\n'%(nu, nv, gntrain))
fl = []
for i in range(b):
    f=open('train%d'%i,'r')
    f.readline()
    fl.append(f)
for i in range(maxntrain):
    for j in range(b):
        if i < lntrain[j]:
            t = fl[j].readline()
  	    t = t.split(',')
	    fw.write('%d\t%d\t%f\n'%(int(t[0]),int(t[1]),float(t[2])))
for i in range(b):
    fl[i].close()
fw.close()

fw=open('wiki_graphchi_test','w')
fw.write('%%MatrixMarket matrix coordinate real general\n%d %d %d\n'%(nu, nv, gntest))
fl = []
for i in range(b):
    f=open('test%d'%i,'r')
    f.readline()
    fl.append(f)
for i in range(maxntest):
    for j in range(b):
        if i < lntest[j]:
            t = fl[j].readline()
  	    t = t.split(',')
	    fw.write('%d\t%d\t%f\n'%(int(t[0]),int(t[1]),float(t[2])))
for i in range(b):
    fl[i].close()
fw.close()

print 'graphchi done\n'

#pre_for_protobuf
fr = open('wiki_graphchi_train','r')
fr.readline()
fr.readline()
for i in range(b):
    fw=open('wiki_train%d'%i,'w')
    if i == b-1:
        fw.write('%d\n'%(gntrain-i*(gntrain/b)))
        for j in range(gntrain-i*(gntrain/b)):
            t = fr.readline()
   	    t=t.split()
            fw.write('%d,%d,%f\n'%(int(t[0]),int(t[1]), float(t[2])))
    else:
        fw.write('%d\n'%(gntrain/b))
        for j in range(gntrain/b):
            t = fr.readline()
   	    t=t.split()
            fw.write('%d,%d,%f\n'%(int(t[0]),int(t[1]), float(t[2])))
    	
    fw.close()
 	    
fr.close()
