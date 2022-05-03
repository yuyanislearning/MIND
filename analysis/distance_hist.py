import numpy as np
import os
import matplotlib.pyplot as plt

adj_dir = '/workspace/PTM/Data/Musite_data/Structure/pdb/AF_cont_map/'

his = []
for fle in os.listdir(adj_dir):
    if fle.endswith('npy'):
        adj = np.load(adj_dir+fle)
        x,y = np.where(adj>0)
        sub_his = abs(x-y)
        sub_his = sub_his[sub_his>30]
        his.append(sub_his)

his = np.concatenate(his,axis=0)


np.save('./distance_histogram.npy', his)
n, bins, patches = plt.hist(x=his, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of spatially close aa distance on sequence')
# plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()
plt.savefig('/workspace/PTM/PTM-pattern-finder/analysis/dist.png')