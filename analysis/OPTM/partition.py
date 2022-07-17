import json
import math
import random
import pdb

with open('/workspace/PTM/Data/OPTM/combined/all.json') as f:
    dat = json.load(f)

uniref_uid = {}
count = 0
with open('/workspace/PTM/Data/OPTM/combined/all_uniref.tab') as f:
    for line in f:
        if count ==0:
            count+=1
            continue
        line = line.strip().split('\t')
        uniref_uid[line[1]] = line[0].split(',')

uniref = list(uniref_uid.keys())

n = len(uniref)

# get uniref id
uids = [u for uni in uniref_uid for u in uniref_uid[uni] ]
uids = list(set(uids))

test_size = math.floor(n/10)
test_id = random.sample(list(range(n)), n)
test_uniref = [ uniref[i] for i in test_id[0:test_size]]
val_uniref = [ uniref[i]  for i in test_id[test_size:test_size*2] ]
train_uniref = [ uniref[i] for i in test_id[test_size*2:n] ]


test_set_uid = [j for i in test_uniref for j in uniref_uid[i]]
test_set = {}
for i in test_set_uid:
    # print(dat[i])
    if dat.get(i,-1)!=-1:
        test_set[i] = dat[i]

val_set_uid = [j for i in val_uniref for j in uniref_uid[i]]
val_set = {}
for i in val_set_uid:
    if dat.get(i,-1)!=-1:
        val_set[i] = dat[i]

train_set_uid = [j for i in train_uniref for j in uniref_uid[i]]
train_set = {}
for i in train_set_uid:
    if dat.get(i,-1)!=-1:
        train_set[i] = dat[i]

with open('/workspace/PTM/Data/OPTM/combined/PTM_test.json','w') as fw:
    json.dump(test_set, fw)


with open('/workspace/PTM/Data/OPTM/combined/PTM_train.json','w') as fw:
    json.dump(train_set, fw)


with open('/workspace/PTM/Data/OPTM/combined/PTM_val.json','w') as fw:
    json.dump(val_set, fw)