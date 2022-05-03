import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


embs = np.load('../temp_pad.npy')
kinases = np.load('../temp_kin.npy')
fig_dir = './figures/kinase_saliency_pad/'


all_st_kinase = ['p38_Kin','CDK1_1','PLK1','Casn_Kin2','PKC_epsilon','DNA_PK','AuroA','PKC_zeta',\
    'PKC_common','GSK3_Kin','ATM_Kin','AMPK','PKA_Kin','Clk2_Kin','Cdc2_Kin','Cam_Kin2','GSK3b',\
        'Erk1_Kin','CDK1_2','Casn_Kin1','Akt_Kin','AuroB','PKC_delta','Cdk5_Kin','PKC_mu']
acid_kinase = ['Casn_Kin1','Casn_Kin2','GSK3_Kin','GSK3b','PLK1']
base_kinase = ['Akt_Kin','Cam_Kin2','Clk2_Kin','PKA_Kin','PKC_epsilon','PKC_zeta','PKC_common',\
    'PKC_delta','PKC_mu','AuroA','AuroB','AMPK']
DNA_damage_kinase = ['ATM_Kin','DNA_PK']
pro_depend_kinase = ['Cdc2_Kin', 'Cdk5_Kin', 'Erk1_Kin', 'p38_Kin', 'CDK1_1', 'CDK1_2']
all_y_kinase = ['EGFR_Kin','Fgr_Kin','Lck_Kin','Src_Kin','InsR_Kin','PDGFR_Kin','Itk_Kin','Abl_Kin']
all_kinase = all_st_kinase + all_y_kinase

for kin in all_kinase:
    select_index = [k for k,kinase in enumerate(kinases) if kinase==kin]
    kin_emb = embs[select_index]
    kin_emb = kin_emb - np.expand_dims(np.mean(kin_emb, axis=1),1)
    kin_emb = kin_emb / np.expand_dims(np.std(kin_emb, axis=1),1)
    kin_emb = np.mean(kin_emb, axis=0)
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    ax.plot(list(range(len(kin_emb))), kin_emb)
    ax.scatter(10, kin_emb[10], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    plt.show()
    plt.savefig(fig_dir+kin+'.png')
    plt.close()