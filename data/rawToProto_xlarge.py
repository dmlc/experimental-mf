import os
from random import shuffle
########### input ########
b=10
raw_data = 'yahoo_raw_train'
userwise_data = 'yahoo_userwise_train_split%d'%b
###########################

fr = open(raw_data,'r')
nr = int(fr.readline())
for i in range(b):
    f=open('raw%d'%i,'w')
    if i == b-1:
	tt = nr - i*(nr/b)
	f.write('%d\n'%(tt))
        for j in range(tt):
	    line = fr.readline()
	    l = line.split(',')
	    u = int(l[0])
	    v = int(l[1])
	    r = float(l[2])
	    f.write('%d,%d,%f\n'%(u,v,r))
    else:
	f.write('%d\n'%(nr/b))
	for j in range(nr/b):
	    line = fr.readline()
	    l = line.split(',')
	    u = int(l[0])
	    v = int(l[1])
	    r = float(l[2])
	    f.write('%d,%d,%f\n'%(u,v,r))
    f.close()
fr.close()
print 'split raw done\n'

for i in range(b):
    data = []
    f=open('raw%d'%i,'r')
    fw=open('raw_shuffle%d'%i,'w')
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

fl = []
fn = []
for i in range(b):
    f=open('raw_shuffle%d'%i,'r')
    nn = int(f.readline())
    fn.append(nn)
    fl.append(f)
fw=open(userwise_data, 'w')
for i in range(b):
    du={}
    for j in range(b):
	if i == b-1:
	    for k in range(fn[j]/b+fn[j]%b):
		li = fl[j].readline().split(',')
		u=int(li[0])
		v=int(li[1])
		r=float(li[2])
		if u in du:
		    du[u].append((v,r))
		else:
		    du[u]=[]
		    du[u].append((v,r))
	else:
	    for k in range(fn[j]/b):
		li = fl[j].readline().split(',')
		u=int(li[0])
		v=int(li[1])
		r=float(li[2])
		if u in du:
		    du[u].append((v,r))
		else:
		    du[u]=[]
		    du[u].append((v,r))
    for u in du:
	fw.write('%d:\n'%u)
	for (v,r) in du[u]:
	    fw.write('%d,%f\n'%(v,r))
for i in range(b):
    fl[i].close()
fw.close()

