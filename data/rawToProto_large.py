from random import shuffle

b=10

fr = open('yahoo_raw_train', 'r')
fw = open('yahoo_pre_train_%dblocks'%b, 'w')

line = fr.readline()
nn = int(line)
data = []
for i in range(nn):
    line = fr.readline()
    tmp = line.split(',')
    uid = int(tmp[0])
    vid = int(tmp[1])
    rating = float(tmp[2])
    data.append((uid,vid,rating))
fr.close()

shuffle(data)
if b!=1:
    shuffle(data)
    shuffle(data)

x = len(data)/b

for i in range(b):
    print i
    f=open('%d'%i,'w')
    if i == (b-1):
	for ele in data[i*x:]:
	    uid = ele[0]
	    vid = ele[1]
	    r = ele[2]
	    f.write('%d,%d,%f\n'%(uid,vid,r))
    else:
	for ele in data[i*x:(i+1)*x]:
	    uid = ele[0]
	    vid = ele[1]
	    r = ele[2]
	    f.write('%d,%d,%f\n'%(uid,vid,r))
    f.close()

for i in range(b):
    print i
    du = {}
    f=open('%d'%i, 'r')
    lines = f.readlines()
    for l in lines:
	t = l.split(',')
	uid = int(t[0])
	vid = int(t[1])
	r = float(t[2])
	if uid in du:
	    du[uid].append((vid,r))
	else:
	    du[uid] = []
	    du[uid].append((vid,r))
    for u in du:
	fw.write('%d:\n'%u)
	for (v,ra) in du[u]:
	    fw.write('%d,%f\n'%(v,ra))
	    
fw.close()
