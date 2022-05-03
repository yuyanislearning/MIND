import json
import pdb

with open('/workspace/PTM/Data/Musite_data/PTM_test/human_OPTM.json') as f:
    dat = json.load(f)

fw = open('/workspace/PTM/PTM-pattern-finder/analysis/OPTM/human_OPTM_res.txt','w')
fw2 = open('/workspace/PTM/PTM-pattern-finder/analysis/OPTM/human_OPTM_res_0.8.txt','w')

for k in dat:
    uid = k.split('|')[1]
    site = str(int(k.split('_')[2])+1)
    ptm = k.split('_')[3]+ k.split('_')[4]
    ttt = k.split('|')[2]
    gene_name = ttt.split('_')[0]
    if float(dat[k]) > 0.5:
        fw.write('\t'.join([uid,gene_name, site, ptm, str(dat[k])])+ '\n')
    if float(dat[k]) > 0.8:
        fw2.write('\t'.join([uid,gene_name, site, ptm, str(dat[k])])+ '\n')
fw.close()
fw2.close()
