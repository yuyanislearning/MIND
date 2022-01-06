import numpy as np
import json
from tqdm import tqdm

file_name = '/workspace/PTM/Data/Musite_data/ptm/PTM_test.json'
with open(file_name, 'r') as fp:
    # data structure: {PID:{seq, label:[{site, ptm_type}]}}
    dat = json.load(fp)

label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}

label_pn = {label:{'P':0,'N':0} for label in label2aa}

for label in label2aa:
    for k in tqdm(dat):
        if dat[k].get('seq',-1)==-1:
            continue
        sequence = str(dat[k]['seq'])
        labels = dat[k]['label']
        P_exist = False
        for lbl in labels:
            if lbl['ptm_type']==label:
                P_exist=True
                label_pn[label]['P']+=1
            else:
                continue
        if P_exist:
            label_pn[label]['N']+= sum([s in label2aa[label] for s in sequence])

label_pn = {label:label_pn[label]['P']/(label_pn[label]['P']+label_pn[label]['N']) for label in label_pn}

print(label_pn)
