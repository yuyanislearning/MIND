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

from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, TransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

model_name = 'saved_model/LSTMTransformer/LSTMTransformer_514_multin_layer_3_15_fold_random'
fold = 15
avg_weight = False


# change it here
class temp_flag():
    def __init__(self, seq_len=514, d_model=128, batch_size=64, model='Transformer',\
         neg_sam=False, dat_aug=False, dat_aug_thres=None, ensemble=False, random_ensemble=False, embedding=False, n_fold=None):
        self.eval = True
        self.seq_len = seq_len
        self.graph = False
        self.fill_cont = None
        self.d_model = d_model
        self.batch_size = batch_size
        self.model = model
        self.neg_sam = neg_sam
        self.dat_aug = dat_aug
        self.dat_aug_thres = dat_aug_thres
        self.ensemble = ensemble
        self.random_ensemble = random_ensemble
        self.embedding = embedding
        self.n_fold = n_fold

def predict(model,seq_len,aug, batch_size, unique_labels, binary=False):
    # predict cases
    ptm_type = {i:p for i, p in enumerate(unique_labels)}

    if binary:# TODO add or remove binary
        y_trues = []
        y_preds = []
    else:
        y_trues = {ptm_type[i]:[] for i in ptm_type}#{ptm_type:np.array:(n_sample,1)}
        y_preds = {ptm_type[i]:[] for i in ptm_type}

    for test_X,test_Y,test_sample_weights in aug:
        y_pred = model.predict(test_X, batch_size=batch_size)
        # seq_len = test_X[0].shape[1]
        if not binary:
            y_mask = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
            y_true = test_Y.reshape(-1, seq_len, len(unique_labels))
            y_pred = y_pred.reshape(-1, seq_len, len(unique_labels))
            for i in range(len(unique_labels)):
                y_true_i = y_true[:,:,i]
                y_pred_i = y_pred[:,:,i]
                y_mask_i = y_mask[:,:,i]

                y_true_i = y_true_i[y_mask_i==1]
                y_pred_i = y_pred_i[y_mask_i==1]
                y_trues[ptm_type[i]].append(y_true_i)
                y_preds[ptm_type[i]].append(y_pred_i)
        else:
            y_mask = test_sample_weights
            y_true = test_Y
    y_trues = {ptm:np.concatenate(y_trues[ptm],axis=0) for ptm in y_trues}
    y_preds = {ptm:np.concatenate(y_preds[ptm],axis=0) for ptm in y_preds}
                
    return y_trues, y_preds    

def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[str(i)][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


def cut_protein(sequence, seq_len):
    # cut the protein if it is longer than chunk_size
    # only includes labels within middle chunk_size//2
    # during training, if no pos label exists, ignore the chunk
    # during eval, retain all chunks for multilabel; retain all chunks of protein have specific PTM for binary
    chunk_size = seq_len - 2
    assert chunk_size%4 == 0
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    records = []
    if len(sequence) > chunk_size:
        for i in range((len(sequence)-1)//half_chunk_size):
            # the number of half chunks=(len(sequence)-1)//chunk_size+1,
            # minus one because the first chunks contains two halfchunks
            max_seq_ind = (i+2)*half_chunk_size
            if i==0:
                cover_range = (0,quar_chunk_size*3)
            elif i==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size, len(sequence)-i*half_chunk_size)
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
            seq = sequence[i*half_chunk_size: max_seq_ind]
            # idx = [j for j in range(len((seq))) if (seq[j] in aa and j >= cover_range[0] and j < cover_range[1])]
            records.append({
                'chunk_id': i,
                'seq': seq,
                # 'idx': idx
            })
    else:
        records.append({
            'chunk_id': 0,
            'seq': sequence,
            # 'idx': [j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records



def main(argv):
    FLAGS = temp_flag()
    limit_gpu_memory_growth()


    label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
    labels = list(label2aa.keys())
    # get unique labels
    unique_labels = sorted(set(labels))
    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
    chunk_size = FLAGS.seq_len - 2

    model = tf.keras.models.load_model(model_name+'_fold_0')
    
    seq = 'RHKDESTNQCGPAVILMFYW'
    print(len(seq))

    X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
    X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
    
    emb_model = keras.models.Model(
        [model.inputs], [model.get_layer('embedding').output]
    )
    emb = emb_model(X)
    emb = emb[0, 1:21,:].numpy()
    labels = np.array(['Positive']*3 + ['Negative']*2 + ['Polar uncharged']*4 + ['Special']*3+['hydrophobic']*8)
    pca_plot(emb, labels, 'res/AA_emb/PCA.pdf', seq)

# def create_baseline(seq_len):
#     # create an all padding sequence
#     return np.array(seq_len * [additional_token_to_index['<PAD>']])



def tsne_plot(embs, true_labels, fig_path, perplexity=30):
    X_embedded = TSNE(perplexity=perplexity).fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'Category':true_labels})
    sns.scatterplot(data=dat, x='X', y='Y', hue='Category', s=30, palette='Set1')
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
