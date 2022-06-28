import json
import pdb
import pprint

ptm_type = ['Hydro_K','Hydro_P','Methy_K','Methy_R','N6-ace_K','Palm_C','Phos_ST','Phos_Y',
'Pyro_Q','SUMO_K','Ubi_K','glyco_N','glyco_ST']
ptm_type =  ["Arg-OH_R","Asn-OH_N","Asp-OH_D","Cys4HNE_C","CysSO2H_C",\
    "CysSO3H_C","Lys-OH_K","Lys2AAA_K","MetO_M","MetO2_M","Phe-OH_F",\
        "ProCH_P","Trp-OH_W","Tyr-OH_Y","Val-OH_V"] #+ ptm_type 


dat_dir = '/workspace/PTM/Data/OPTM/'

for case in ['train', 'val', 'test']:
    d = {}
    for p in ptm_type:
        d[p] = 0

    with open(dat_dir + '/PTM_'+case+'.json') as f:
        dat = json.load(f)
    for uid in dat:
        for s in dat[uid]['label']:
            d[s['ptm_type']]+=1
    print(case)
    pprint.pprint(d)

for case in ['train', 'val', 'test']:
    d = {}
    for p in ptm_type:
        d[p] = {}
    with open(dat_dir + '/PTM_'+case+'.json') as f:
        dat = json.load(f)

    for uid in dat:
        for j in dat[uid]['label']:
            aa_site = dat[uid]['seq'][j['site']]
            if d[j['ptm_type']].get(aa_site,-1)==-1:
                d[j['ptm_type']][aa_site] = 1
            else:
                d[j['ptm_type']][aa_site]+= 1
    aa_dist = [(k,d[k]) for k in d.keys()]

    print(case)
    print('aa_dist')
    pprint.pprint(aa_dist)            