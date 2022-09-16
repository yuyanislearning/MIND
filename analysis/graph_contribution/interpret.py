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

import sys
import pdb

sys.path.append('/local2/yuyan/PTM-Motif/PTM-pattern-finder/')
from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator, handle_flags
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, LSTMTransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

bthre = 0.3
uthre = 1-bthre
model_name = '/local2/yuyan/PTM-Motif/PTM-pattern-finder/saved_model/con_g'

handle_flags()


def interpolate_emb_specific( emb, alphas, seq_idx, baseline_med='blank', baseline=None, nbs = None):
    # interpolate embedding and baseline
    if baseline is None:
        baseline = get_baseline_specific(emb, baseline_med, nbs)
        # baseline = np.zeros(emb.shape)
        if baseline is None:
            return None, None
    # else:
    #     baseline = get_baseline(emb, baseline_med, seq_idx, baseline)
    #     if baseline is None:
    #         return None, None
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline = tf.cast(baseline, emb.dtype)
    emb_x = tf.expand_dims( emb, 0)
    baseline_x = tf.expand_dims(baseline, 0)
    emb_x = tf.tile(emb_x, (1,len(nbs),1,1))# match the 512 aa selected in baseline
    delta = emb_x - baseline_x #(1, 512, seq_len, dim)
    embs = baseline_x + alphas_x * delta #(alpha, 512, seq_len, dim)
    seq_len, dim = embs.shape[2], embs.shape[3]
    embs = tf.reshape(embs, (len(alphas)*len(nbs), seq_len, dim))# reshape to batch first
    # baseline_x = tf.reshape(baseline_x, (len(alphas)*21, seq_len, dim))
    return embs, baseline


def get_baseline_specific(emb, baseline_med, nbs, baseline=None):
    if baseline_med =='blank':
        tile_emb = tf.tile(emb, (len(nbs),1,1)).numpy() # duplicate the batch
        for i,nb in enumerate(nbs):
            tile_emb[i,nb,:] = 0 # set as zero for specific aa
        
    # elif baseline_med == 'pad':
    #     tile_emb = tf.tile(emb, (21,1,1)).numpy()
    #     baseline = tf.tile(baseline, (21,1,1)).numpy()
    #     for i in range(21):
    #         tile_emb[i, seq_idx-10+i, :] = baseline[i, seq_idx-10+i, :] # replace with pad baseline
        
    tile_emb = tf.convert_to_tensor(tile_emb)
    return tile_emb

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
            idx = [j for j in range(len((seq))) if (seq[j] in aa and j >= cover_range[0] and j < cover_range[1])]
            records.append({
                'chunk_id': i,
                'seq': seq,
                'idx': idx
            })
    else:
        records.append({
            'chunk_id': None,
            'seq': sequence,
            'idx': [j for j in range(len((sequence))) if sequence[j] in aa]
        })
    return records



def get_gradients(X, emb_model,  grad_model, top_pred_idx, seq_idx, embedding=None, method=None, emb=None, baseline=None, graph=None, nbs=None):
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
        # test is pred is > pred_thres
        
        # batching since it's too big
        alpha, seq_len, dim = embedding.shape[0]//len(nbs), embedding.shape[1],embedding.shape[2]
        embedding = tf.reshape(embedding, (alpha, len(nbs), seq_len, dim))
        final_grads = []
        for i in range(len(nbs)):
            with tf.GradientTape() as tape:
                embed = embedding[:,i,:,:] #(alpha,)
                tape.watch(embed)
                temp_X = [ tf.tile(x, tf.constant([alpha]+(len(x.shape)-1)*[1])) for x in X] + [embed] # tile sequence x to match emb
                out_pred = grad_model(temp_X)
                out_pred = tf.reshape(out_pred, (51, -1, 13))
                top_class = out_pred[:,seq_idx, top_pred_idx]
                

            grads = tape.gradient(top_class, embed) # (alpha, seq, dim)
            grads = (grads[:-1] + grads[1:]) / tf.constant(2.0) # calculate integration
            integrated_grads = tf.math.reduce_mean(grads, axis = 0) * (emb[i,:,:] - baseline[i,:,:])  # integration
            final_grads.append(tf.reduce_sum(integrated_grads[ nbs[i], :], axis=-1).numpy()) #norm of the specific aa

        return np.array(final_grads)


def build_model_graph(FLAGS, optimizer , unique_labels, pretrain_model):
    if FLAGS.model=='LSTMTransformer':
        model = LSTMTransFormer(FLAGS,FLAGS.model,optimizer,  \
            num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.1,binary=False,\
            unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
        model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
        for layer in pretrain_model.layers:
            if len(layer.get_weights())!=0 and layer.name!='my_last_dense' and layer.name!='embedding':
                if layer.name=='encoder_layer':
                    model.model.get_layer(name='encoder_layer_0').set_weights(layer.get_weights())
                else:
                    model.model.get_layer(name=layer.name).set_weights(layer.get_weights())   

    print(model.model.summary())
    return model.model

def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]


def heatmap(grads, nbs,highlight_idx, fle):
    
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    ax.plot(nbs, grads)
    ax.scatter(highlight_idx, grads[np.where(nbs==highlight_idx)], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    ax.set_xlabel(nbs)
    # ax = sns.heatmap(a)
    
    # sns.lineplot(list(range(len(a))), a)
    # plt.plot(highlight_idx, a[highlight_idx], markersize=29, fillstyle='none', markeredgewidth=1.5)
    plt.show()
    plt.savefig(fle)
    plt.close()
    



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

    with open('true_diff_res.json') as f:
        dat = json.load(f)    

    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=FLAGS.learning_rate, amsgrad=True)



    model = tf.keras.models.load_model(model_name)

    grad_model = build_model_graph(FLAGS, optimizer , unique_labels, model)

    emb_model = keras.models.Model(
            [model.inputs], [model.get_layer('embedding').output]
        )

        

    # emb_model = keras.models.Model(
    #     [models[0].inputs], [models[0].get_layer('encoder_layer').output]
    # )
    
    
    for uid in dat:
        for pred in dat[uid]:
            if pred[2]>uthre and pred[3]< bthre:
                with open('/local2/yuyan/PTM-Motif/Data/Musite_data/fasta/'+uid+'.fa') as fr:
                    fa = list(SeqIO.parse(fr, 'fasta'))[0]
                sequence = str(fa.seq)
                uid = str(fa.id)
                if '|' in uid:
                    uid = uid.split('|')[1]
                records = cut_protein(sequence, FLAGS.seq_len, 'STY')#label2aa[FLAGS.label] 

                for record in records:
                    seq = record['seq']
                    idx = record['idx']
                    chunk_id = record['chunk_id']
                    if chunk_id is None:
                        name_id = ''
                        chunk_id = 0
                    else:
                        name_id = '~'+str(chunk_id)


                    adj = np.load('../../ttt/'+uid+name_id+'_514_5.npy')
                    adj = np.expand_dims(adj, 0)
                    X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
                    X = [tf.expand_dims(X, 0),adj, tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
                    
                    if chunk_id==0:
                        cover_range = (0,chunk_size//4*3)
                    elif chunk_id==((len(sequence)-1)//-1):
                        cover_range = (chunk_size//4+chunk_id*chunk_size//2, len(sequence))
                    else:
                        cover_range = (chunk_size//4+chunk_id*chunk_size//2, chunk_size//4+(chunk_id+1)*chunk_size//2)

                    site = pred[0]
                    if site+1 >=cover_range[0] and site+1< cover_range[1]:  
                        seq_idx = site - chunk_id*chunk_size//2 
                        # get intergrated gradient for specific ptm
                        
                        temp_label = pred[1]
                        emb = emb_model(X)
                        m_steps = 50
                        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
                        nbs = np.where(adj[0,seq_idx+1,]>0)[0]
                        if max(nbs)-min(nbs)<20:
                            continue
                        interpolated_emb, baseline = interpolate_emb_specific(emb, alphas, seq_idx+1, nbs=nbs)
                        if baseline is None:
                            continue
                        emb_grad = get_gradients(X, emb_model, grad_model, label_to_index[temp_label], \
                            seq_idx+1, interpolated_emb, method='integrated_gradient', emb=tf.tile(emb,(len(nbs),1,1)), baseline=baseline, graph = adj, nbs=nbs)

                        heatmap(emb_grad, nbs,seq_idx+1, '/local2/yuyan/PTM-Motif/PTM-pattern-finder/analysis/graph_contribution/figs/'+uid+'_'+str(pred[0])+'_'+pred[1]+'.pdf')
                

if __name__ == '__main__':
    app.run(main)
      
