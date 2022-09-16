import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import pdb
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


# os.system('rm /local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/*')
embs = np.load('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp_0.8.npy')
kins = np.load('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp_kin_0.8.npy')
seqs = np.load('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp_seq_0.8.npy')


all_st_kinase = ['p38_Kin','CDK1_1','PLK1','Casn_Kin2','PKC_epsilon','DNA_PK','AuroA','PKC_zeta',\
    'PKC_common','GSK3_Kin','ATM_Kin','AMPK','PKA_Kin','Clk2_Kin','Cdc2_Kin','Cam_Kin2','GSK3b',\
        'Erk1_Kin','CDK1_2','Casn_Kin1','Akt_Kin','AuroB','PKC_delta','Cdk5_Kin','PKC_mu']
acid_kinase = ['Casn_Kin1','Casn_Kin2','GSK3_Kin','GSK3b','PLK1']
base_kinase = ['Akt_Kin','Cam_Kin2','Clk2_Kin','PKA_Kin','PKC_epsilon','PKC_zeta','PKC_common',\
    'PKC_delta','PKC_mu','AuroA','AuroB','AMPK']
DNA_damage_kinase = ['ATM_Kin','DNA_PK']
pro_depend_kinase = ['Cdc2_Kin', 'Cdk5_Kin', 'Erk1_Kin', 'p38_Kin', 'CDK1_1', 'CDK1_2']
all_y_kinase = ['EGFR_Kin','Fgr_Kin','Lck_Kin','Src_Kin','InsR_Kin','PDGFR_Kin','Itk_Kin','Abl_Kin']

group_kinases = []
select_index = list(range(len(kins)))
for i,kin in enumerate(kins):
    if kin in acid_kinase:
        group_kinases.append('Acidophilic')
    elif kin in base_kinase:
        group_kinases.append('Basophilic')
    # elif kin in DNA_damage_kinase:
    #     group_kinases.append('DNA damage')
    elif kin in pro_depend_kinase:
        group_kinases.append('Proline-dependent')
    # elif kin in all_y_kinase:
    #     group_kinases.append('Phos on tyrosine')
    else:
        select_index.remove(i)

kins = kins[select_index]
embs = embs[select_index]
seqs = seqs[select_index]
n_clus = 17



def plot(emb,fle):
    emb = np.abs(emb)
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    ax.plot(list(range(-7,8,1)), emb)
    ax.scatter(0, emb[7], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    # ax = sns.heatmap(a)
    
    # sns.lineplot(list(range(len(a))), a)
    # plt.plot(highlight_idx, a[highlight_idx], markersize=29, fillstyle='none', markeredgewidth=1.5)
    plt.show()
    plt.savefig(fle)
    plt.close()


def tsne_plot(embs, true_labels, fig_path, perplexity=100):
    col = sns.color_palette("Set2", 8)+sns.color_palette("Set1", 9)
    X_embedded = TSNE(perplexity=perplexity,  random_state=100).fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'kinase':true_labels})
    sns.scatterplot(data=dat, x='X', y='Y', hue='kinase', s=10, palette=col)
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()

distortions = []
for n_clus in range(5, 30):
    kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(embs)
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(16,8))
plt.plot(list(range(5,30)), distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
plt.savefig('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/elbow.pdf')
plt.close()
pdb.set_trace()
kmeans = KMeans(n_clusters=n_clus, random_state=0).fit(embs)


# tsne_plot(embs, group_kinases, '/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/tsne_group.pdf', perplexity=100)
# tsne_plot(embs, kmeans.labels_, '/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/tsne_17.pdf', perplexity=100)
# pdb.set_trace()

for i in range(n_clus):
    plot(kmeans.cluster_centers_[i][3:18], '/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/'+str(i)+'.pdf')
    fw = open('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/seq_'+str(i)+'.txt','w')
    fw2 = open('/local2/yuyan/PTM-Motif/PTM-pattern-finder/saliency/dat/temp/kin_'+str(i)+'.txt','w')
    for j,k in zip(seqs[kmeans.labels_==i], kins[kmeans.labels_==i]):
        if len(j)==21:
            fw.write(j[3:18]+'\n')
            fw2.write(k+'\n')
            
    fw.close()
    fw2.close()

