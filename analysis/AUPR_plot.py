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
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, average_precision_score
import pandas as pd

import json
sys.path.append('../')

import pdb
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_class_weights,  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

def eval(model, seq_len,aug, batch_size, unique_labels,graph=False, binary=False, ind=None, num_cont=None):
    # aug = PTMDataGenerator( test_data, seq_len,model=self.model_name, batch_size=batch_size, unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont, d_model=self.d_model, eval=True)#test_data.Y.shape[0]
    # aug = PTMDataGenerator( test_data, seq_len, model=self.model_name,batch_size=len(aug.list_id)//5+1, unique_labels=unique_labels, graph = graph,shuffle=False, binary=binary, ind=ind, num_cont=num_cont, d_model=self.d_model, eval=True)#test_data.Y.shape[0]
    # test_X, test_Y, test_sample_weights = aug.__getitem__(0)
    ptm_type = {i:p for i, p in enumerate(unique_labels)}

    if binary:
        y_trues = []
        y_preds = []
    else:
        y_trues = {ptm_type[i]:[] for i in ptm_type}
        y_preds = {ptm_type[i]:[] for i in ptm_type}
    count=1
    for test_X,test_Y,test_sample_weights in aug:
        count+=1
        y_pred = model.predict(test_X, batch_size=batch_size)
        # seq_len = test_X[0].shape[1]
        if not binary:
            y_mask_all = test_sample_weights.reshape(-1, seq_len, len(unique_labels))
            y_true_all = test_Y.reshape(-1, seq_len, len(unique_labels))
            y_pred_all = y_pred.reshape(-1, seq_len, len(unique_labels))
        else:
            y_mask = test_sample_weights
            y_true = test_Y
                

        AUC = {}
        PR_AUC = {}
        confusion_matrixs = {}
        if binary:
            y_true = y_true[y_mask==1]
            y_pred = y_pred[y_mask==1]
            y_trues.append(y_true)
            y_preds.append(y_pred)
        else:
            for i in range(len(unique_labels)):
                y_true = y_true_all[:,:,i]
                y_pred = y_pred_all[:,:,i]
                y_mask = y_mask_all[:,:,i]

                y_true = y_true[y_mask==1]
                y_pred = y_pred[y_mask==1]
                y_trues[ptm_type[i]].append(y_true)
                y_preds[ptm_type[i]].append(y_pred)

    if binary:
        y_trues = np.concatenate(y_trues, axis=0)
        y_preds = np.concatenate(y_preds, axis=0)
        AUC = roc_auc_score(y_trues, y_preds)
        PR_AUC = average_precision_score(y_trues, y_preds) 
        confusion_matrixs=pd.DataFrame(confusion_matrix(y_trues, y_preds>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
    else:
        y_trues = {ptm:np.concatenate(y_trues[ptm],axis=0) for ptm in y_trues}
        y_preds = {ptm:np.concatenate(y_preds[ptm],axis=0) for ptm in y_preds}
        for i in range(len(unique_labels)):
            # print(y_trues[ptm_type[i]])
            AUC[ptm_type[i]] = roc_auc_score(y_trues[ptm_type[i]], y_preds[ptm_type[i]])
            PR_AUC[ptm_type[i]] = average_precision_score(y_trues[ptm_type[i]], y_preds[ptm_type[i]])
            confusion_matrixs[ptm_type[i]]=pd.DataFrame(confusion_matrix(y_trues[ptm_type[i]], y_preds[ptm_type[i]]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1'])
    
    return AUC, PR_AUC, confusion_matrixs



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


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

limit_gpu_memory_growth()

test_name = '/workspace/PTM/Data/Musite_data/ptm/PTM_test.json'
model_dir = '/workspace/PTM/PTM-pattern-finder/saved_model/'
model_names = [
    'LSTMTransformer/LSTMTransformer_514_negsam_False_binary',
    'LSTMTransformer/LSTMTransformer_514_multin_layer_3'
]
label_model_names = [
    'Binary',
    'Multilabel'
]

FLAGS = temp_flag()
test_dat_aug = PTMDataGenerator(test_name, FLAGS, shuffle=True,ind=None, eval=True)
unique_labels = test_dat_aug.unique_labels

with open('res/why_multilabel.csv','w') as fw:
    to_write = ','.join(['experiment_name']+unique_labels)
    fw.write(to_write+'\n')
    for i in range(len(model_names)):
        model = tf.keras.models.load_model(model_dir+model_names[i])
        _, PR_AUC, _ = eval(model, FLAGS.seq_len,test_dat_aug, FLAGS.batch_size, unique_labels,graph=False, binary=False, ind=None, num_cont=None)
        to_write = ','.join([label_model_names[i]]+['%.3f'%(PR_AUC[u]) for u in unique_labels])
        fw.write(to_write+'\n')
