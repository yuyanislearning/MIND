import pdb
import json
import numpy as np

in_dir = '/local2/yuyan/PTM-Motif/Data/Musite_data/PTM_test/eval/'

with open(in_dir+'MIND.json') as f:
    MIND_dat = json.load(f)
with open(in_dir+'con_g.json') as f:
    g_dat = json.load(f)
    

with open('/local2/yuyan/PTM-Motif/Data/Musite_data/ptm/PTM_test.json') as f:#
    dat = json.load(f)


ptm2ptm = {'O-linked_glycosylation':'glyco_ST', 'S-palmitoyl_cysteine':'Palm_C','Hydroxyproline':'Hydro_P',\
    'Pyrrolidone_carboxylic_acid':'Pyro_Q','Phosphoserine':'Phos_ST', 'Hydroxylysine':'Hydro_K',\
    'Ubiquitination':'Ubi_K','Methyllysine':'Methy_K','N6-acetyllysine':'N6-ace_K',\
    'SUMOylation':'SUMO_K','Methylarginine':'Methy_R','Phosphotyrosine':'Phos_Y',\
    'N-linked_glycosylation':'glyco_N','Phosphothreonine':'Phos_ST'}


label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}



MIND_preds = {}
for sptm in MIND_dat:
    uid = sptm.split('_')[0]
    site = int(sptm.split('_')[1])
    ptm_type = sptm.split('_')[2]+'_'+sptm.split('_')[3]
    if MIND_preds.get(uid,-1)==-1:
        MIND_preds[uid] = [(site, ptm_type, float(MIND_dat[sptm]))]
    else:
        MIND_preds[uid].append((site, ptm_type, float(MIND_dat[sptm])))


g_preds = {}
for sptm in g_dat:
    uid = sptm.split('_')[0]
    site = int(sptm.split('_')[1])
    ptm_type = sptm.split('_')[2]+'_'+sptm.split('_')[3]
    if g_preds.get(uid,-1)==-1:
        g_preds[uid] = [(site, ptm_type, float(g_dat[sptm]))]
    else:
        g_preds[uid].append((site, ptm_type, float(g_dat[sptm])))

diff_pred = {}
for uid in g_preds:
    g_pred = g_preds[uid]
    MIND_pred = MIND_preds[uid]
    for i in range(len(g_pred)):
        assert g_pred[i][0] == MIND_pred[i][0]
        if (g_pred[i][2]>0.5 and MIND_pred[i][2] < 0.5) or (g_pred[i][2]<0.5 and MIND_pred[i][2] > 0.5):
            if diff_pred.get(uid, -1)==-1:
                diff_pred[uid] = [(g_pred[i][0], g_pred[i][1], g_pred[i][2], MIND_pred[i][2])]
            else:
                diff_pred[uid].append((g_pred[i][0], g_pred[i][1], g_pred[i][2], MIND_pred[i][2]))
true_diff_pred = {k:[] for k in diff_pred}

for uid in diff_pred:
    for pred in diff_pred[uid]:
        sites = [lbl['site'] for lbl in dat[uid]['label'] if lbl['ptm_type']==pred[1]]
        if pred[2]>0.5:# case of con_g predicting true
            if pred[0] in sites:# predict correct 
                true_diff_pred[uid].append(pred)
        else:
            if pred[0] not in sites:
                true_diff_pred[uid].append(pred)

count=0
for uid in diff_pred:
    for pred in diff_pred[uid]:
        count+=1
print(count)
count=0
for uid in true_diff_pred:
    for pred in true_diff_pred[uid]:
        count+=1
print(count)

with open('/local2/yuyan/PTM-Motif/PTM-pattern-finder/analysis/graph_contribution/true_diff_res.json','w') as f:
    json.dump(true_diff_pred, f)