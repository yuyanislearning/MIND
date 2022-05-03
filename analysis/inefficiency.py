import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import pdb


with open('/workspace/PTM/Data/Musite_data/ptm/all.json') as f:
    dat = json.load(f)

label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
        'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}

my = []
them = []

window_size = 31

for uid in dat:
    seq = dat[uid]['seq']
    label = dat[uid]['label']
    if len(label)==0:
        continue
    ptms = set([l['ptm_type'] for l in label])
    total_count = 0
    for ptm in ptms:
        aas = label2aa[ptm]
        n_label = len([s for s in seq if s in aas])
        total_count+=n_label
    my.append(len(seq))
    them.append(total_count*window_size)

# my = [math.log2(m) for m in my]
# them = [math.log2(tt) for tt in them]

my = pd.DataFrame(data={'# amino acid':my, 'type':'whole sequence'})
them = pd.DataFrame(data={'# amino acid':them, 'type':'sliding window'})
final_dat = pd.concat([my, them], ignore_index=True)
plt.ylim(0, 8000)

sns.boxplot(x=final_dat['type'], y=final_dat['# amino acid'], fliersize=0)

plt.show()
plt.savefig('figures/inefficient_boxplot.png')