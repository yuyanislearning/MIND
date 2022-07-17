import json

# combined the total json
with open('/workspace/PTM/Data/Musite_data/ptm/all.json') as f:
    all = json.load(f)

with open('/workspace/PTM/Data/OPTM/OPTM_filtered.json') as f:
    optm = json.load(f)

print(len(all))
for uid in optm:
    if all.get(uid, -1)==-1:
        all[uid] = optm[uid]
    else:
        all[uid]['label'].extend(optm[uid]['label'])

print(len(all))
with open('/workspace/PTM/Data/OPTM/combined/all.json','w') as f:
    json.dump(all, f)

# get all pro
with open('/workspace/PTM/Data/OPTM/combined/all_pro.txt','w') as f:
    for k in all:
        f.write(k+'\n')


# run partition

