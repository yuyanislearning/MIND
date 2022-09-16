import numpy as np
import os.path
from os.path import exists
import pdb
import json
import os
import matplotlib.pyplot as plt


dir = '/local2/yuyan/PTM-Motif/Data/Musite_data/Structure/pdb/AF_updated_alpha_shape'
with open('/local2/yuyan/PTM-Motif/Data/Musite_data/ptm/all.json') as f:
    dat = json.load(f)
a = os.popen('ls '+ dir )
coverage = []
not_cover = []
n_covers = []
covers = []
n_ncs = []

for line in a:
    filename = line.strip()
    uid = filename.split('_')[0]
    adj = np.load(os.path.join(dir, filename), allow_pickle=True)
    diag = np.zeros(adj.shape)
    np.fill_diagonal(diag, 1)
    adj = adj - diag

    cover_ls = np.where(np.sum(adj, axis=0)>0)[0]
    sites = [lbl['site'] for lbl in dat[uid]['label']]
    cover_sites = [site in cover_ls for site in sites]
    coverage.append(len(cover_ls)/adj.shape[0])
    covers.append(len(cover_ls))
    not_cover.append(adj.shape[0]-len(cover_ls))
    n_covers.append(np.sum(cover_sites))
    n_ncs.append(len(sites) - np.sum(cover_sites))
    # print('total length %d'%(adj.shape[0]))
    # print('cover aa: %d'%(len(cover_ls)))
    # print('cover ptms: %d'%(np.sum(cover_sites)))
    # print('ptm not covered: %d'%(len(sites) - np.sum(cover_sites)))

coverage = np.array(coverage)
# n, bins, patches = plt.hist(coverage, 50, density=False, facecolor='b', alpha=0.75)
# plt.show()
# plt.savefig('temp_cover.png')
# plt.close()

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


covers = np.array(covers)
not_cover = np.array(not_cover)
not_cover+= 1
n_covers = np.array(n_covers)
n_ncs = np.array(n_ncs)
print('total number cover: %d'%(np.sum(n_covers)))
print('total number not cover: %d'%(np.sum(n_ncs)))

norm_cover = n_covers/covers
norm_na =  n_ncs/not_cover
columns = [norm_cover, norm_na]
fig, ax = plt.subplots()
ax.violinplot(columns, showextrema=False)
labels = ['# normalized PTM in alpha shape', '# normalized not in']
set_axis_style(ax, labels)
ax.set(ylim=(0, 0.1))
plt.show()

plt.savefig('temp.png')
plt.close()

