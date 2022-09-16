import json
from os.path import exists
import pdb

with open('/local2/yuyan/PTM-Motif/Data/Musite_data/ptm/PTM_train.json') as f:
    dat = json.load(f)

print(len(dat))
to_pop = []
for uid in dat:
    # pdb.set_trace()
    if not exists('/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated/AF-'+uid+'-F1-model_v3.cif'):
        to_pop.append(uid)

for uid in to_pop:
    dat.pop(uid)
print(len(dat))

with open('/local2/yuyan/PTM-Motif/Data/Musite_data/ptm/ptm_w_stru/PTM_train.json','w') as fw:
    json.dump(dat, fw)
