#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from datetime import datetime
from tqdm import tqdm
from pprint import pprint

import json
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def dist_plot(embs, fig_path):
    embs = np.sum(embs, 0)
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    ax.plot(list(range(len(embs))), embs)
    ax.scatter(10, embs[10], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()

embs = np.load('temp_pad.npy')
kinases = np.load('temp_kin.npy')
all_st_kinase = ['p38_Kin','CDK1_1','PLK1','Casn_Kin2','PKC_epsilon','DNA_PK','AuroA','PKC_zeta',\
    'PKC_common','GSK3_Kin','ATM_Kin','AMPK','PKA_Kin','Clk2_Kin','Cdc2_Kin','Cam_Kin2','GSK3b',\
        'Erk1_Kin','CDK1_2','Casn_Kin1','Akt_Kin','AuroB','PKC_delta','Cdk5_Kin','PKC_mu']
acid_kinase = ['Casn_Kin1','Casn_Kin2','GSK3_Kin','GSK3b','PLK1']
base_kinase = ['Akt_Kin','Cam_Kin2','Clk2_Kin','PKA_Kin','PKC_epsilon','PKC_zeta','PKC_common',\
    'PKC_delta','PKC_mu','AuroA','AuroB','AMPK']
DNA_damage_kinase = ['ATM_Kin','DNA_PK']
pro_depend_kinase = ['Cdc2_Kin', 'Cdk5_Kin', 'Erk1_Kin', 'p38_Kin', 'CDK1_1', 'CDK1_2']
all_y_kinase = ['EGFR_Kin','Fgr_Kin','Lck_Kin','Src_Kin','InsR_Kin','PDGFR_Kin','Itk_Kin','Abl_Kin']
# all st kinase

for kina in all_st_kinase:
    select_index = [k for k,kin in enumerate(kinases) if kin in kina]
    select_embs = embs[select_index,]
    select_kinases = kinases[select_index,]
    dist_plot(select_embs, 'analysis/figures/kinase_dist_pad/'+kina+'.png')
    # good shape: ATM_kin