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

from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, TransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

# model_name = 'saved_model/LSTMTransformer/LSTMTransformer_514_multin_layer_3_15_fold_random'#
model_name = 'saved_model/LSTMTransformer/LSTMTransformer_514_multin_layer_3OPTM_r15'
fold = 15


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


def cut_protein(sequence, seq_len, seq_idx):
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
                cover_range = (quar_chunk_size+i*half_chunk_size, len(sequence))
                max_seq_ind = len(sequence)
            else:
                cover_range = (quar_chunk_size+i*half_chunk_size, quar_chunk_size+(i+1)*half_chunk_size)
            seq = sequence[i*half_chunk_size: max_seq_ind]
            if seq_idx is not None:
                if seq_idx >= cover_range[0] and seq_idx < cover_range[1]:
                    records.append({
                        'chunk_id': i,
                        'seq': seq,
                        'idx': seq_idx - i*chunk_size//2
                    })
                    break
            else:
                records.append({'chunk_id':i,
                'seq':seq})
    else:
        if seq_idx is not None:
            records=[{
            'chunk_id': 0,
            'seq': sequence,
            'idx': seq_idx
        }]
        else:
            records = [{'chunk_id':0,
            'seq':sequence}]
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

    FLAGS = temp_flag()
    limit_gpu_memory_growth()

    # label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    # 'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
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
    with open(model_name+'_PRAU.json') as f:
        AUPR_dat = json.load(f)
    
    models = [] # load models
    for i in range(fold):
        models.append(tf.keras.models.load_model(model_name+'_fold_'+str(i)))#
    # model = models[0]
    # emb_model = keras.models.Model(
    #     [models[0].inputs], [models[0].get_layer('encoder_layer').output]
    # )
    weights = ensemble_get_weights(AUPR_dat, unique_labels)
    # SNPs = {'O94759':[('V','934','I')], 'P02649':[('C','130','R')], 'Q5S007':[('G','2019','S')],
    # 'Q8TF76':[('V','76','E')],'Q96RQ3':[('H','464','P')],'Q9BSA9':[('M','393','T')],'Q9C0K1':[('A','391','T')]}
    for uid in ['Q5S007']:#SNPs:
        y_preds = {}
        y_preds_mut = {}
        # snps = SNPs[uid]
        snps = [("M","712","V"),("R","793","M"),("Q","930","R"),("R","1067","Q"),("S","1096","C"),
        ("I","1122","V"),("S","1228","T"),("I","1371","V"),("R","1441","C"),("R","1441","G"),("R","1441","H"),
        ("R","1514","Q"),("P","1542","S"),("V","1598","E"),("Y","1699","C"),("R","1728","H"),("R","1728","L"),
        ("M","1869","T"),("R","1941","H"),("I","2012","T"),("G","2019","S"),("I","2020","T"),("T","2031","S"),
        ("T","2141","M"),("R","2143","H"),("D","2175","H"),("Y","2189","C"),("T","2356","I"),("G","2385","R"),
        ("V","2390","M"),("L","2439","I"),("L","2466","H")]
        with open('/workspace/PTM/Data/Musite_data/fasta/'+uid+'.fa') as ffast:
            sequence = str(list(SeqIO.parse(ffast, 'fasta'))[0].seq)
        # sequence = dat[uid]['seq']
        for snp in snps:
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

                X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
                X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
                
                for j in range(fold):#fold
                    y_pred = models[j].predict(X)#*weights[ptm][j]                    
                    y_pred = y_pred.reshape(1, FLAGS.seq_len, -1)
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
                        cover_range = (quar_chunk_size, len(sequence)-i*half_chunk_size)
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

                X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
                X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
                
                for j in range(fold):#fold
                    y_pred = models[j].predict(X)#*weights[ptm][j]                    
                    y_pred = y_pred.reshape(1, FLAGS.seq_len, -1)
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
                        cover_range = (quar_chunk_size, len(sequence)-i*half_chunk_size)
                    else:
                        cover_range = (quar_chunk_size, quar_chunk_size+half_chunk_size)
                    idx = [j for j in range(len((seq))) if (seq[j] in label2aa[ptm] and j >= cover_range[0] and j < cover_range[1])]

                    for i in idx:
                        ix = i+chunk_id*(FLAGS.seq_len-2)//2 +1
                        y_preds_mut[str(ix)+'_'+ptm] = str(y_pred_sum[0, i+1,label_to_index[ptm]])

            with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_'+SNP_wt+str(SNP_site)+SNP_var+'_OPTM.json','w') as fw:
                json.dump(y_preds_mut, fw)
        with open('/workspace/PTM/Data/PTMVar/res/'+uid+'_OPTM.json','w') as fw:
            json.dump(y_preds, fw)

# def main(argv):

#     FLAGS = temp_flag()
#     limit_gpu_memory_growth()

#     label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
#     'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
#     labels = list(label2aa.keys())
#     # get unique labels
#     unique_labels = sorted(set(labels))
#     label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
#     index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}
#     chunk_size = FLAGS.seq_len - 2

#     with open('/workspace/PTM/Data/PTMVar/all_PTMVar.json') as f:
#         PTMVar = json.load(f)

#     with open('/workspace/PTM/Data/Musite_data/ptm/all.json') as f:
#         dat = json.load(f)

#     with open(model_name+'_PRAU.json') as f:
#         AUPR_dat = json.load(f)
    
#     models = [] # load models
#     for i in range(fold):
#         models.append(tf.keras.models.load_model(model_name+'fold_'+str(i)))

#     # emb_model = keras.models.Model(
#     #     [models[0].inputs], [models[0].get_layer('encoder_layer').output]
#     # )
#     weights = ensemble_get_weights(AUPR_dat, unique_labels)

#     fw = open('./analysis/res/SNP_PTM_updated.tsv','w')
#     for uid in tqdm(PTMVar): # for every fasta contains phos true label
#         for var in PTMVar[uid]:
#             if not exists('/workspace/PTM/Data/Musite_data/fasta/'+uid+'.fa'):
#                 continue
#             with open('/workspace/PTM/Data/Musite_data/fasta/'+uid+'.fa') as ffast:
#                 sequence = str(list(SeqIO.parse(ffast, 'fasta'))[0].seq)
#             # sequence = dat[uid]['seq']
#             SNP_index = int(var[1])-1
#             SNP_sequence = sequence[:SNP_index] + var[2] + sequence[(SNP_index + 1):]
#             if var[0] != sequence[int(var[1])-1]:
#                 continue
#             if var[4] != sequence[int(var[3])-1]:
#                 continue
#             PTM_index = int(var[3])-1
#             records = cut_protein(sequence, FLAGS.seq_len, PTM_index)#label2aa[FLAGS.label] 

#             seq = records[0]['seq']
#             idx = records[0]['idx']
#             chunk_id = records[0]['chunk_id']

#             X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
#             X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
            
#             for j in range(fold):#fold
#                 y_pred = models[j].predict(X)#*weights[ptm][j]                    
#                 y_pred = y_pred.reshape(1, FLAGS.seq_len, -1)
#                 temp_weight = np.array([weights[p][j] for p in weights])
#                 y_pred = y_pred*temp_weight#TODO 
#                 if j==0:
#                     y_pred_sum = y_pred
#                 else:
#                     y_pred_sum += y_pred

#             y_pred_sum = y_pred_sum.reshape(1, -1, 13)
#             pred_prob = y_pred_sum[0,idx+1,label_to_index[var[5]]]
#             thres = 0.5
#             if pred_prob > thres:
#                 records = cut_protein(SNP_sequence, FLAGS.seq_len, PTM_index)#label2aa[FLAGS.label] 
#                 seq = records[0]['seq']
#                 idx = records[0]['idx']
#                 chunk_id = records[0]['chunk_id']
#                 X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
#                 X = [tf.expand_dims(X, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
#                 for j in range(fold):#fold
#                     y_pred = models[j].predict(X)#*weights[ptm][j]                    
#                     y_pred = y_pred.reshape(1, FLAGS.seq_len, -1)
#                     temp_weight = np.array([weights[p][j] for p in weights])
#                     y_pred = y_pred*temp_weight#TODO 
#                     if j==0:
#                         y_pred_sum = y_pred
#                     else:
#                         y_pred_sum += y_pred
#                 y_pred_sum = y_pred_sum.reshape(1, -1, 13)
#                 SNP_pred_prob = y_pred_sum[0,idx+1,label_to_index[var[5]]]
#                 if SNP_pred_prob<0.5:
#                     fw.write('\t'.join([uid, var[0], var[1], var[2], var[3], var[4], var[5],str(pred_prob), str(SNP_pred_prob)])+'\n')
#     fw.close()

def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


if __name__ == '__main__':
    app.run(main)
