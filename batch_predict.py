#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from tqdm import tqdm
from pprint import pprint
import pdb

import json
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
from os.path import exists

from src.utils import get_class_weights, handle_flags, assign_neighbour, rm_diag, pad_adj
from src import utils
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token

OPTM=False
handle_flags()


if not OPTM:
    label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
else:
    label2aa = {"Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
        "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
        "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
    

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
            elif i==((len(sequence)-1)//half_chunk_size):
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
            'chunk_id': None,
            'seq': sequence,
            # 'idx': [j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records


def main(argv):
    FLAGS = flags.FLAGS

    # tf.config.run_functions_eagerly(True)

    labels = list(label2aa.keys())
    # get unique labels
    unique_labels = sorted(set(labels))
    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
    chunk_size = FLAGS.seq_len - 2

    with open(FLAGS.pretrain_name +'_PRAU.json') as f:
        AUPR_dat = json.load(f)
    
    # models = [tf.keras.models.load_model(model_name)]
    models = [] # load models
    for i in range(FLAGS.n_fold):#
        models.append(tf.keras.models.load_model(FLAGS.pretrain_name+'_fold_'+str(i)))
    # model = tf.keras.models.load_model(model_name)

    weights = ensemble_get_weights(AUPR_dat, unique_labels)

    y_preds = {}
    chunk_size = FLAGS.seq_len - 2
    quar_chunk_size = chunk_size//4
    half_chunk_size = chunk_size//2

    # for ptm in label2aa.keys():
    # with open('/workspace/PTM/Data/OPTM/pig_prot_all.fasta', 'r') as fp:#TODO # /workspace/PTM/Data/BCAA/mouse_pdh.fasta
        # for rec in tqdm(SeqIO.parse(fp, 'fasta')): # for every fasta contains phos true label
        #     sequence = str(rec.seq)
        #     uid = str(rec.id)

    # for i in range(1):#place holder
    # with open('/workspace/PTM/Data/OPTM/nonoverlap_uid.txt') as f:    
    #     for line in tqdm(f):
    #             uid = line.strip()
    #         sequence = dat[uid]['seq']
    y_preds = {}
    seqs = []
    chunk_ids = []
    uids = []
    pad_Xs = []
    adjs = []
    sequences = []
    count=0
    with open(FLAGS.data_path, 'r') as fp:
        dat = list(SeqIO.parse(fp, 'fasta'))            
    for dat_count, rec in tqdm(enumerate(dat)):
        seqid = rec.id
        if len(seqid.split('|')) >= 2:
            uid = seqid.split('|')[1]
        else:
            uid = seqid
        sequence=str(rec.seq)
        records = cut_protein(sequence, FLAGS.seq_len)
        rec_count = 0
        for record in records:
            count+=1
            rec_count+=1
            seq = record['seq']
            chunk_id = record['chunk_id']
            seqs.append(seq)
            chunk_ids.append(chunk_id)
            X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
            pad_Xs.append(X)
            uids.append(uid)
            sequences.append(sequence)
            if FLAGS.graph:
                get_graph(uid,X,  FLAGS)
                id = uid + '~' + str(chunk_id) if len(records)>1 else uid
                adj = np.load('./temp/'+id+'_'+str(FLAGS.seq_len)+'_5'+'.npy', allow_pickle=True) 
                adjs.append(adj)
            if count<FLAGS.batch_size and dat_count<len(dat)-1:
                continue
            pad_Xs = np.stack(pad_Xs, axis=0)
            X = [pad_Xs]
            if FLAGS.graph:
                adjs = np.stack(adjs, axis=0)
                X.append(adjs)
            
            batch_size = pad_Xs.shape[0]
            X.append(np.zeros((batch_size, 514, 128)))
            
            for j in range(FLAGS.n_fold):#fold 
                y_pred = models[j](X)#*weights[ptm][j]                    
                y_pred = tf.reshape(y_pred, (batch_size, FLAGS.seq_len, -1)).numpy()
                temp_weight = np.array([weights[p][j] for p in weights])
                y_pred = y_pred*temp_weight#TODO 
                if j==0:
                    y_pred_sum = y_pred
                else:
                    y_pred_sum += y_pred

            for ptm in label2aa.keys():
                for ch in range(batch_size):
                    if chunk_ids[ch]==0:#TODO
                        cover_range = (0,quar_chunk_size*3)
                    elif chunk_ids[ch]==((len(sequences[ch])-1)//half_chunk_size-1):
                        cover_range = (quar_chunk_size, len(seqs[ch]))
                    elif chunk_ids[ch] is None:
                        cover_range = (0,515)
                        chunk_ids[ch] = 0
                    else:
                        cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
                    idx = [j for j in range(len((seqs[ch]))) if (seqs[ch][j] in label2aa[ptm] and j >= cover_range[0] and j < cover_range[1])]
                    # idx = [j for j in range(len((seq))) if seq[j] in label2aa[ptm]]
                    for i in idx:
                        ix = i+chunk_ids[ch]*(FLAGS.seq_len-2)//2
                        y_preds[(str(uids[ch]), str(ix+1), ptm)] = str(y_pred_sum[ch, i+1,label_to_index[ptm]])

            seqs = []
            chunk_ids = []
            uids = []
            pad_Xs = []
            adjs = []
            sequences = []
            count=0

    with open(os.path.join(FLAGS.res_path,'result.json'),'w') as fw:
        json.dump({f"{a}_{b}_{c}": v for (a, b, c), v in y_preds.items()}, fw)
    
    correct_pred = {k: v for k, v in y_preds.items() if float(v) >= 0.5}
    correct_df_data = [list(k) + [v] for k, v in correct_pred.items()]
    correct_df = pd.DataFrame(correct_df_data, columns =['uid','site','PTM_type','pred_score'])
    correct_df = correct_df.sort_values(by=['pred_score'], ascending=False)
    correct_df.to_csv(os.path.join(FLAGS.res_path,'correct_predictions.csv'), index=False)
    
def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


def get_graph(uid,X,  FLAGS):
    print('constructing graphs')
    adj_name = './temp/'+uid+'_'+str(FLAGS.seq_len)+'_'+str(FLAGS.fill_cont)+'.npy'
    if not exists(adj_name):
        if '~' in uid:
            n_seq = int(uid.split('~')[1])
            tuid = uid.split('~')[0]

            if exists(FLAGS.adj_dir+tuid+'.cont_map.npy'):
                adj = np.load(FLAGS.adj_dir+tuid+'.cont_map.npy')
                n = adj.shape[0]
                left_slice = n_seq*FLAGS.chunk_size//2
                right_slice = min((n_seq+2)*FLAGS.chunk_size//2, n)
                adj = adj[left_slice:right_slice, left_slice:right_slice]
            else:
                n = np.where(np.array(X)==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, FLAGS.fill_cont)
        else:
            if exists(FLAGS.adj_dir+uid+'.cont_map.npy'):
                adj = np.load(FLAGS.adj_dir+uid+'.cont_map.npy')
                adj = rm_diag(adj,FLAGS.fill_cont)
            else:
                # 24 is the stop sign
                n = np.where(np.array(X)==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, FLAGS.fill_cont)
        adj = pad_adj(adj, FLAGS.seq_len)
        
        np.save(adj_name,adj)
    else:
        next
        # try:
        #     adj = np.load(adj_name)
        # except:
        #     os.system('rm '+ adj_name)

    return None

if __name__ == '__main__':
    app.run(main)
