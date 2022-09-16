import os
import json
import pdb

a = os.popen('ls /local2/yuyan/PTM-Motif/Data/Musite_data/PTM_test/BCAA/').read().split('\n')
for i in a:
    uid = i.split('.json')[0]
    f = open('/local2/yuyan/PTM-Motif/Data/Musite_data/PTM_test/temp/'+uid+'_OPTM.txt','w')
    with open('/local2/yuyan/PTM-Motif/Data/Musite_data/PTM_test/BCAA/'+i) as fr:
        dat = json.load(fr)
    for k in dat:
        if float(dat[k])>0.5:
            prob = dat[k]
            k = k.split('_')
            f.write('\t'.join([k[0],str(int(k[1])+1),k[2],k[3],prob])+'\n')
    f.close()
