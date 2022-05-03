import numpy as np
import json
from tqdm import tqdm
from pprint import pprint
import pdb
import matplotlib.pyplot as plt


for ttt in ['train','val','test']:
    file_name = '/workspace/PTM/Data/Musite_data/ptm/PTM_' + ttt+ '.json'
    prot_length = []
    with open(file_name, 'r') as fp:
        # data structure: {PID:{seq, label:[{site, ptm_type}]}}
        dat = json.load(fp)
    for k in dat:
        prot_length.append(len(dat[k]['seq']))

prot_length = np.array(prot_length)

np.save('./res/prot_length.npy', prot_length)
n, bins, patches = plt.hist(x=prot_length, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of protein lengths')
# plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.axvline(x=512)
plt.show()
plt.savefig('/workspace/PTM/PTM-pattern-finder/analysis/figures/prot_length_dist.png')