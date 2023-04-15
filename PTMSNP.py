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
import copy
from os.path import exists

import json
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
import pandas as pd
import re
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


import pdb

from src.utils import get_class_weights,  limit_gpu_memory_growth, handle_flags
from src import utils
from src.model import TransFormerFixEmbed
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

OPTM = False
handle_flags() 

def ensemble_get_weights(PR_AUCs, unique_labels):
    weights = {ptm:None for ptm in unique_labels}
    for ptm in unique_labels:
        weight = np.array([PR_AUCs[str(i)][ptm] for i in range(len(PR_AUCs))])
        weight = weight/np.sum(weight)
        weights[ptm] = weight
    return weights # {ptm_type}


def cut_protein(sequence, seq_len, aa):
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
                'idx': None
            })
    else:
        records.append({
            'chunk_id': None,
            'seq': sequence,
            'idx': None#[j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records



def get_gradients(X, emb_model,  grad_model, top_pred_idx, seq_idx, embedding=None, method=None, emb=None, baseline=None):
    """Computes the gradients of outputs w.r.t input embedding.

    Args:
        embedding: input embedding
        top_pred_idx: Predicted label for the input image
        seq_idx: location of the label

    Returns:
        Gradients of the predictions w.r.t embedding
    """

    if method == 'gradient':
        embedding = emb_model(X)

        with tf.GradientTape() as tape:
            tape.watch(embedding)
            temp_X = X + [embedding]
            out_pred = grad_model(temp_X)
            top_class = out_pred[0,seq_idx, top_pred_idx] 

        grads = tape.gradient(top_class, embedding)        
        return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(grads), axis = -1)).numpy()

    if method == 'integrated_gradient':
        with tf.GradientTape() as tape:
            tape.watch(embedding)
            temp_X = [ tf.tile(x, tf.constant([embedding.shape[0]]+(len(x.shape)-1)*[1])) for x in X] + [embedding]
            out_pred = grad_model(temp_X)
            top_class = out_pred[:,seq_idx, top_pred_idx]
            

        grads = tape.gradient(top_class, embedding)
        grads = (grads[:-1] + grads[1:]) / tf.constant(2.0)
        return tf.math.sqrt(tf.reduce_mean(tf.math.square(tf.math.reduce_mean(grads, axis = 0) * (emb - baseline)), axis=-1)).numpy(), top_class[-1]



# for single protein single SNP all ptm
def main(argv):
    FLAGS = flags.FLAGS
    limit_gpu_memory_growth()

    if not OPTM:
        label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
        'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
    else:
        label2aa = {"Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
            "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
            "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
    labels = list(label2aa.keys())
    # get unique labels
    unique_labels = sorted(set(labels))
    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
    chunk_size = FLAGS.seq_len - 2
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2
    with open(FLAGS.pretrain_name+'_PRAU.json') as f:
        AUPR_dat = json.load(f)
    
    models = [] # load models
    for i in range(FLAGS.n_fold):
        models.append(tf.keras.models.load_model(FLAGS.pretrain_name+'_fold_'+str(i)))#

    weights = ensemble_get_weights(AUPR_dat, unique_labels)
    ftr = open(FLAGS.data_path)
    count=0
    with open(FLAGS.data_path, 'r') as fp:
        rec = list(SeqIO.parse(fp, 'fasta'))[0]
            
    uid = rec.id.split('|')[1]
    sequence=str(rec.seq)        
    snp = FLAGS.snp.split('_')
    
    y_preds = {}
    y_preds_mut = {}
    
    # sequence = dat[uid]['seq']
    SNP_site = int(snp[1])
    SNP_var = snp[2]
    SNP_wt = snp[0]
    SNP_index = SNP_site-1
    assert sequence[SNP_index]==SNP_wt
    SNP_sequence = sequence[:SNP_index] + SNP_var + sequence[(SNP_index + 1):]
    
    records = cut_protein(sequence, FLAGS.seq_len, None)#label2aa[FLAGS.label] 

    for record in records:
        seq = record['seq']
        chunk_id = record['chunk_id']
        if chunk_id is None:
            name_id = ''
            chunk_id = 0
        else:
            name_id = '~'+str(chunk_id)

        # adj = np.load('./ttt/'+uid+name_id+'_514_5.npy')
        # adj = np.expand_dims(adj, 0)
        X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
        X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]

        for j in range(FLAGS.n_fold):#fold
            y_pred = models[j](X)#*weights[ptm][j]                    
            y_pred = y_pred.numpy().reshape(1, FLAGS.seq_len, -1)
            temp_weight = np.array([weights[p][j] for p in weights])
            y_pred = y_pred*temp_weight#TODO 
            if j==0:
                y_pred_sum = y_pred
            else:
                y_pred_sum += y_pred

        for ptm in label2aa.keys():
            if chunk_id==0:
                cover_range = (0,quar_chunk_size*3)
            elif chunk_id==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size, len(sequence)-chunk_id*half_chunk_size)
            else:
                cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
            idx = [j for j in range(len((seq))) if (seq[j] in label2aa[ptm] and j >= cover_range[0] and j < cover_range[1])]
            for i in idx:
                ix = i+chunk_id*(FLAGS.seq_len-2)//2+1
                y_preds[str(ix)+'_'+ptm] = str(y_pred_sum[0, i+1,label_to_index[ptm]])

    records = cut_protein(SNP_sequence, FLAGS.seq_len, None)#label2aa[FLAGS.label] 

    for record in records:
        seq = record['seq']
        chunk_id = record['chunk_id']
        if chunk_id is None:
            name_id = ''
            chunk_id = 0
        else:
            name_id = '~'+str(chunk_id)

        # adj = np.load('./ttt/'+uid+name_id+'_514_5.npy')
        # adj = np.expand_dims(adj, 0)
        X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
        X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
        
        for j in range(FLAGS.n_fold):#fold
            y_pred = models[j](X)#*weights[ptm][j]                    
            y_pred = y_pred.numpy().reshape(1, FLAGS.seq_len, -1)
            temp_weight = np.array([weights[p][j] for p in weights])
            y_pred = y_pred*temp_weight#TODO 
            if j==0:
                y_pred_sum = y_pred
            else:
                y_pred_sum += y_pred

        for ptm in label2aa.keys():
            if chunk_id==0:
                cover_range = (0,quar_chunk_size*3)
            elif chunk_id==((len(sequence)-1)//half_chunk_size-1):
                cover_range = (quar_chunk_size, len(sequence)-chunk_id*half_chunk_size)
            else:
                cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
            idx = [j for j in range(len((seq))) if (seq[j] in label2aa[ptm] and j >= cover_range[0] and j < cover_range[1])]

            for i in idx:
                ix = i+chunk_id*(FLAGS.seq_len-2)//2 +1
                y_preds_mut[str(ix)+'_'+ptm] = str(y_pred_sum[0, i+1,label_to_index[ptm]])
    with open(os.path.join(FLAGS.res_path, uid+'_'+SNP_wt+str(SNP_site)+SNP_var+'.json'),'w') as fw:
        json.dump(y_preds_mut, fw)
    with open(os.path.join(FLAGS.res_path, uid+'.json'),'w') as fw:
        json.dump(y_preds, fw)


def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


if __name__ == '__main__':
    app.run(main)
