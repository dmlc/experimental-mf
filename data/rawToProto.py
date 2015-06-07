from random import shuffle

b=1

fr = open('raw_test_data', 'r')
fw = open('netflix_pre_test_%dblocks'%b, 'w')

line = fr.readline()
lines = fr.readlines()
fr.close()
data = []
for line in lines:
    tmp = line.split(',')
    uid = int(tmp[0])
    vid = int(tmp[1])
    rating = float(tmp[2])
    data.append((uid,vid,rating))

shuffle(data)
if b!=1:
    shuffle(data)
    shuffle(data)

x = len(data)/b
for i in range(b):
    du = {}
    if i == (b-1):
        for ele in data[i*x:]:
	    uid = ele[0]
	    vid = ele[1]
	    r = ele[2]
 	    if uid in du:
	    	du[uid].append((vid,r))
	    else:
	    	du[uid] = []
	    	du[uid].append((vid,r))
  	for u in du:
	    fw.write('%d:\n'%u)
	    for (v,ra) in du[u]:
		fw.write('%d,%f\n'%(v,ra))
    else:
	for ele in data[i*x:(i+1)*x]:
	    uid = ele[0]
	    vid = ele[1]
	    r = ele[2]
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
