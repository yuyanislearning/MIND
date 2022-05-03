import json
import pdb

with open('/workspace/PTM/Data/Musite_data/PTM_test/PTM_on_OPTMb.json') as f:
    dat = json.load(f)

with open('/workspace/PTM/Data/OPTM/OPTM_filtered.json') as f:
    optm = json.load(f)

pred = {}
for k in dat:
    if float(dat[k])<0.5:
        continue
    uid = k.split('_')[0]
    site = int(k.split('_')[1])
    ptm = k.split('_')[2] +'_'+ k.split('_')[3]
    if ptm != 'Hydro_K':
        continue
    if pred.get(uid,-1)==-1:
        pred[uid] = [(site,ptm, dat[k])]
    else:
        pred[uid].append((site,ptm, dat[k]))
    
optm_count = 0
find_count = 0
with open('/workspace/PTM/Data/OPTM/nonoverlap_uid.txt') as f:
    for line in f:
        uid = line.strip()
        for lab in optm[uid]['label']:
            if lab['ptm_type']=='Lys-OH_K': 
                # print(uid)
                # print(lab)
                # if pred.get(uid,-1)==-1:
                #     continue
                # print(pred[uid])
                optm_count+=1
                if pred.get(uid,-1)==-1:
                    continue
                if any([sit[0]==lab['site'] for sit in pred[uid]]):
                    find_count+=1
                
print(optm_count)
print(find_count)


with open('/workspace/PTM/Data/OPTM/pred_hydro_k.txt','w') as f:
    for k in pred:
        for dd in pred[k]:
            f.write('\t'.join([k, str(dd[0]),dd[1],dd[2] ])+'\n')