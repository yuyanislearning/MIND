import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pdb


def tsne_plot(embs, true_labels, fig_path, perplexity=30):
    X_embedded = TSNE(perplexity=perplexity).fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'kinase':true_labels})
    col = sns.color_palette("Set1")
    ttt = set(true_labels)
    new_col = col[0:3]+[col[4]]
    sns.scatterplot(data=dat, x='X', y='Y', hue='kinase', s=10, palette=new_col)
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    

def pca_plot(embs, kinases, fig_path):
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'kinase':kinases})
    sns.scatterplot(data=dat, x='X', y='Y', hue='kinase', s=10, palette='Set1')
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()    

def thres(emb, thres=5):
    # retain top thres 
    new_emb = np.copy(emb)
    zero_out = thres - emb.shape[0]
    new_emb[np.argpartition(-new_emb,zero_out)[zero_out:]]=0
    new_emb[10] = 0 # remove center
    return new_emb


embs = np.load('saliency/dat/temp_0.8.npy')
kinases = np.load('saliency/dat/temp_kin_0.8.npy')
fig_dir = 'analysis/figures/tsne_emb_0.8/'

all_st_kinase = ['p38_Kin','CDK1_1','PLK1','Casn_Kin2','PKC_epsilon','DNA_PK','AuroA','PKC_zeta',\
    'PKC_common','GSK3_Kin','ATM_Kin','AMPK','PKA_Kin','Clk2_Kin','Cdc2_Kin','Cam_Kin2','GSK3b',\
        'Erk1_Kin','CDK1_2','Casn_Kin1','Akt_Kin','AuroB','PKC_delta','Cdk5_Kin','PKC_mu']
acid_kinase = ['Casn_Kin1','Casn_Kin2','GSK3_Kin','GSK3b','PLK1']
base_kinase = ['Akt_Kin','Cam_Kin2','Clk2_Kin','PKA_Kin','PKC_epsilon','PKC_zeta','PKC_common',\
    'PKC_delta','PKC_mu','AuroA','AuroB','AMPK']
DNA_damage_kinase = ['ATM_Kin','DNA_PK']
pro_depend_kinase = ['Cdc2_Kin', 'Cdk5_Kin', 'Erk1_Kin', 'p38_Kin', 'CDK1_1', 'CDK1_2']
all_y_kinase = ['EGFR_Kin','Fgr_Kin','Lck_Kin','Src_Kin','InsR_Kin','PDGFR_Kin','Itk_Kin','Abl_Kin']

# for all st kinase
# select_index = [k for k,kin in enumerate(kinases) if kin in acid_kinase+base_kinase+pro_depend_kinase]
# select_embs = embs[select_index,]
# select_kinases = kinases[select_index,]

# group_kinases = []
# for kin in select_kinases:
#     if kin in acid_kinase:
#         group_kinases.append('Acidophilic')
#     elif kin in base_kinase:
#         group_kinases.append('Basophilic')
#     # elif kin in DNA_damage_kinase:
#     #     group_kinases.append('DNA damage')
#     elif kin in pro_depend_kinase:
#         group_kinases.append('Proline-dependent')

# norm_embs = np.divide(select_embs,np.linalg.norm(select_embs, axis=1, keepdims=True))
# thres_embs = [thres(emb,3) for emb in norm_embs]
# thres_embs = np.stack(thres_embs)
# for per in [50,100,200]:
#     tsne_plot(norm_embs, group_kinases, fig_dir+'all_ST_kinase_emb_tsne_'+str(per)+'.pdf', perplexity=per)
#     tsne_plot(thres_embs, group_kinases, fig_dir + 'all_ST_kinase_emb_tsne_thres_'+str(per)+'.pdf',perplexity=per)
# pca_plot(norm_embs, group_kinases, fig_dir+'all_ST_kinase_emb_pca.pdf')
# pca_plot(thres_embs, group_kinases, fig_dir+'all_ST_kinase_emb_pca_thres.pdf')

# kinase_name = ['acid_kinase', 'base_kinase','DNA_damage_kinase','pro_depend_kinase','all_y_kinase']
# for i,kinase in enumerate([acid_kinase, base_kinase,DNA_damage_kinase,pro_depend_kinase,all_y_kinase]):
#     # for all st kinase
#     select_index = [k for k,kin in enumerate(kinases) if kin in kinase]
#     select_embs = embs[select_index,]
#     select_kinases = kinases[select_index,]

#     norm_embs = np.divide(select_embs,np.linalg.norm(select_embs, axis=1, keepdims=True))
#     thres_embs = [thres(emb,3) for emb in norm_embs]
#     thres_embs = np.stack(thres_embs)
#     for per in [5, 50, 100, 500]:
#         tsne_plot(norm_embs, select_kinases, fig_dir+kinase_name[i]+'_emb_tsne_'+str(per)+'.pdf', perplexity=per)
#         tsne_plot(thres_embs, select_kinases, fig_dir +kinase_name[i]+ '_emb_tsne_thres_'+str(per)+'.pdf',perplexity=per)
#     pca_plot(norm_embs, select_kinases, fig_dir+kinase_name[i]+'_emb_pca.pdf')
#     pca_plot(thres_embs, select_kinases, fig_dir+kinase_name[i]+'_emb_pca_thres.pdf')


my_kinase = ['AuroB','Casn_Kin2','AMPK','Erk1_Kin']

# for all st kinase
select_index = [k for k,kin in enumerate(kinases) if kin in my_kinase]
select_embs = embs[select_index,]
select_kinases = kinases[select_index,]

norm_embs = np.divide(select_embs,np.linalg.norm(select_embs, axis=1, keepdims=True))
thres_embs = [thres(emb,3) for emb in norm_embs]
thres_embs = np.stack(thres_embs)

for per in [50,100,500]:
    tsne_plot(norm_embs, select_kinases, fig_dir+'my_emb_tsne_'+str(per)+'.pdf', perplexity=per)
    tsne_plot(thres_embs, select_kinases, fig_dir +'my_emb_thres_tsne_'+str(per)+'.pdf',perplexity=per)
pca_plot(norm_embs, select_kinases, fig_dir+'my_emb_pca.pdf')
pca_plot(thres_embs, select_kinases, fig_dir+'my_emb_thres_pca.pdf')

