import json
import pdb
from Bio import SeqIO

all_optm = ["Arg-OH_R","Asn-OH_N","Asp-OH_D","Cys4HNE_C","CysSO2H_C",\
    "CysSO3H_C","Lys-OH_K","Lys2AAA_K","MetO_M","MetO2_M","Phe-OH_F",\
        "ProCH_P","Trp-OH_W","Tyr-OH_Y","Val-OH_V"]

dat = {}
with open('/workspace/PTM/Data/OPTM/optm_pro.fasta') as f:
    for rec in list(SeqIO.parse(f, 'fasta')):
        uid = rec.id.split('|')[1]
        dat[uid] = str(rec.seq)

new_dat = {}
count=0
bad = 0
bad_seq = 0
with open('/workspace/PTM/Data/OPTM/optm_filtered.csv') as f:
    for line in f:
        if count==0:
            count+=1
            continue
        line = line.strip().split(',')
        uid = line[0]
        if line[1]=='NA':
            continue
        site = int(line[1])
        ptm = line[2]
        try:
            seq = dat[uid]
        except:
            print(uid)
            continue
        try:
            aa = seq[site-1]
        except:
            bad_seq+=1
            continue
        ptm += '_'+aa
        if ptm not in all_optm:
            bad+=1
            continue
        if new_dat.get(uid,-1)==-1:
            new_dat[uid] = {'seq':seq, 'label':[{'site':site-1,'ptm_type':ptm}]}
        else:
            new_dat[uid]['label'].append({'site':site-1,'ptm_type':ptm})
        count+=1

print(bad)
print(bad_seq)
with open('/workspace/PTM/Data/OPTM/OPTM_filtered.json','w') as f:
    json.dump(new_dat, f)