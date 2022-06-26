import json
import pdb

with open('/workspace/PTM/Data/OPTM/OPTM_filtered.json') as f:
    dat = json.load(f)

fw = open('/workspace/PTM/Data/OPTM/argOH_flank.txt','w')
count=1
for uid in dat:
    seq = dat[uid]['seq']
    for lab in dat[uid]['label']:
        if lab['ptm_type'] == 'Arg-OH_R':
            try:
                subseq = seq[lab['site']-7:lab['site']+8]
                if len(subseq)!=15:
                    continue
            except:
                continue
            fw.write(subseq+'\n')



fw.close()