import json

with open('/workspace/PTM/Data/Musite_data/PTM_test/BCAA_prediction.json') as f:
    dat = json.load(f)

fw = open('/workspace/PTM/PTM-pattern-finder/analysis/CVD/BCAA_res.txt','w')

for k in dat:
    uid = k.split('|')[1]
    site = str(int(k.split('_')[2])+1)
    ptm = k.split('_')[3]+ k.split('_')[4]
    ttt = k.split('|')[2]
    gene_name = ttt.split('_')[0]
    if uid in ['Q9P0J1','P29803','Q16654','O00330','P08559','Q15119','Q15118','Q8NCN5','A0A024RBX9','O95563','Q9Y5U8']:
        typ = 'PDH'
    else:
        typ = 'BCAA'

    if float(dat[k]) > 0.5:
        fw.write('\t'.join([uid,gene_name, typ, site, ptm, str(dat[k])])+ '\n')

fw.close()
