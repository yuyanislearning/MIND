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


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import pdb

from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator, handle_flags,cut_protein
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, TransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

fold = 15
avg_weight = False

model_name = 'saved_model/g_fifteenfold_'
handle_flags()

 

def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[str(i)][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


def main(argv):
    FLAGS = flags.FLAGS
    limit_gpu_memory_growth()


    label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
    labels = list(label2aa.keys())
    # get unique labels
    unique_labels = sorted(set(labels))
    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
    chunk_size = FLAGS.seq_len - 2
    
    with open(model_name+'PRAU.json') as f:
        AUPR_dat = json.load(f)
    weights = ensemble_get_weights(AUPR_dat, unique_labels)

    seq = 'RHKDESTNQCGPAVILMFYW'

    adj = np.zeros((1,FLAGS.seq_len,FLAGS.seq_len))
    X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
    X = [tf.expand_dims(X, 0),adj, tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
    
    embs = []
    
    for i in range(fold):
        model = tf.keras.models.load_model(model_name+'fold_'+str(i))
        emb_model = keras.models.Model(
            [model.inputs], [model.get_layer('embedding').output]
        )
        emb = emb_model(X)
        emb = emb[0, 1:21,:].numpy()
        embs.append(emb)
        
    emb = np.mean(np.stack(embs,axis=0), axis=0)
    labels = np.array(['Positive']*3 + ['Negative']*2 + ['Polar uncharged']*4 + ['Special']*3+['hydrophobic']*8)
    pca_plot(emb, labels, 'res/AA_emb/PCA.pdf', seq)
    # tsne_plot(emb, labels, 'res/AA_emb/tsne.pdf', seq)

# def create_baseline(seq_len):
#     # create an all padding sequence
#     return np.array(seq_len * [additional_token_to_index['<PAD>']])



def tsne_plot(embs, true_labels, fig_path, seq, perplexity=30):
    X_embedded = TSNE(perplexity=perplexity).fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'Category':true_labels})
    p1 = sns.scatterplot(data=dat, x='X', y='Y', hue='Category', s=30, palette='Set1')
    
    for line in range(0,dat.shape[0]):
        p1.text(dat.X[line]+0.01, dat.Y[line],
        seq[line], horizontalalignment='left', 
        size='medium', color='black', weight='semibold')
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    

def pca_plot(embs, kinases, fig_path, seq):
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'Category':kinases})
    p1 = sns.scatterplot(data=dat, x='X', y='Y', hue='Category', s=30, palette='Set1')

    for line in range(0,dat.shape[0]):
        p1.text(dat.X[line]+0.01, dat.Y[line],
        seq[line], horizontalalignment='left', 
        size='medium', color='black', weight='semibold')

    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()    


def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


if __name__ == '__main__':
    app.run(main)
