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
import pandas as pd


import pdb

from src.utils import get_class_weights,  limit_gpu_memory_growth, handle_flags
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, LSTMTransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding

model_name = 'saved_model/con_g'
fold = 1
prob_thres = 0.8
handle_flags()
ensemble=False


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
    return model   

def interpolate_emb( emb, alphas, seq_idx, baseline_med='blank', baseline=None):
    # interpolate embedding and baseline
    if baseline is None:
        baseline = get_baseline(emb, baseline_med, seq_idx)
        # baseline = np.zeros(emb.shape)
        if baseline is None:
            return None, None
    else:
        baseline = get_baseline(emb, baseline_med, seq_idx, baseline)
        if baseline is None:
            return None, None
    alphas_x = alphas[:, tf.newaxis, tf.newaxis, tf.newaxis]
    baseline = tf.cast(baseline, emb.dtype)
    emb_x = tf.expand_dims( emb, 0)
    baseline_x = tf.expand_dims(baseline, 0)
    emb_x = tf.tile(emb_x, (1,21,1,1))# match the 21 aa selected in baseline
    delta = emb_x - baseline_x #(1, 21, seq_len, dim)
    embs = baseline_x + alphas_x * delta #(alpha, 21, seq_len, dim)
    seq_len, dim = embs.shape[2], embs.shape[3]
    embs = tf.reshape(embs, (len(alphas)*21, seq_len, dim))# reshape to batch first
    # baseline_x = tf.reshape(baseline_x, (len(alphas)*21, seq_len, dim))
    return embs, baseline

def get_baseline(emb, baseline_med, seq_idx, baseline=None):
    if seq_idx-10<0 or seq_idx+11>emb.shape[1]:
        return None
    if baseline_med =='blank':
        tile_emb = tf.tile(emb, (21,1,1)).numpy() # duplicate the batch
        for i in range(21):
            tile_emb[i,seq_idx-10+i,:] = 0 # set as zero for specific aa
        
    elif baseline_med == 'pad':
        tile_emb = tf.tile(emb, (21,1,1)).numpy()
        baseline = tf.tile(baseline, (21,1,1)).numpy()
        for i in range(21):
            tile_emb[i, seq_idx-10+i, :] = baseline[i, seq_idx-10+i, :] # replace with pad baseline
        
    tile_emb = tf.convert_to_tensor(tile_emb)
    return tile_emb


def get_gradients(X, emb_model,  grad_model, top_pred_idx, seq_idx, embedding=None, method=None, emb=None, baseline=None, graph=None):
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
        alpha, seq_len, dim = embedding.shape[0]//21, embedding.shape[1],embedding.shape[2]
        embedding = tf.reshape(embedding, (alpha, 21, seq_len, dim))
        final_grads = []
        for i in range(21):
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
            final_grads.append(tf.reduce_sum(integrated_grads[ seq_idx-10+i, :], axis=-1).numpy()) #norm of the specific aa

        return np.array(final_grads)

def heatmap(a, highlight_idx, fle):
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')

    std = np.std(a, axis=0)
    ttt = np.squeeze(np.mean(a, axis=0))
    ax.plot(list(range(-7,8,1)), ttt)
    ax.scatter(0, ttt[7], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    ax.fill_between(list(range(-7,8,1)), (ttt-std), (ttt+std), color='b', alpha=.1)
        
    plt.show()
    plt.savefig(fle)
    plt.close()
    return a



# def create_baseline(seq_len):
#     # create an all padding sequence
#     return np.array(seq_len * [additional_token_to_index['<PAD>']])

def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]

def make_gradcam_heatmap(model, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def dist_plot(embs, fig_path, thres=3):
    out = []
    for i in range(embs.shape[0]):
        out.append(np.argpartition(embs[i,:],-1*thres)[(-1*thres):])
    out = np.concatenate(out)
    dat = pd.DataFrame(data = {'X':out})
    sns.histplot(dat, x='X', discrete=True)
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()



def main(argv):
    FLAGS = flags.FLAGS
    limit_gpu_memory_growth()

    label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
    'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
    # label2aa = {"Arg-OH_R":'R',"Asn-OH_N":'N',"Asp-OH_D":'D',"Cys4HNE_C":"C","CysSO2H_C":"C","CysSO3H_C":"C",
    #     "Lys-OH_K":"K","Lys2AAA_K":"K","MetO_M":"M","MetO2_M":"M","Phe-OH_F":"F",
    #     "ProCH_P":"P","Trp-OH_W":"W","Tyr-OH_Y":"Y","Val-OH_V":"V"}
    labels = list(label2aa.keys())
    # get unique labels
    unique_labels = sorted(set(labels))
    label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
    index_to_label = {i: str(label) for i, label in enumerate(unique_labels)}

    model = tf.keras.models.load_model(model_name)
    
    emb_model = keras.models.Model(
        [model.inputs], [model.get_layer('embedding').output]
    )
    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=FLAGS.learning_rate, amsgrad=True)
    grad_model = build_model_graph(FLAGS, optimizer , unique_labels, model).model
    

    chunk_size = FLAGS.seq_len - 2

    # with open('/local2/yuyan/PTM-Motif/Data/OPTM/OPTM_filtered.json') as f:
    #     dat = json.load(f)
    with open('/local2/yuyan/PTM-Motif/Data/Musite_data/Phosphorylation_motif/correct_scan_kinase.json') as f:
        scan = json.load(f)

    

    ptm_type = 'Phos_ST'
    for kin_type in ['CDK1_1']:
        seqs = []
        all_saliency = []
        optm_count = 0
        protein_count = 0
        fp =  open('/local2/yuyan/PTM-Motif/Data/Musite_data/Phosphorylation_motif/all_phos_uid.txt', 'r')
        for line in tqdm(fp): # for every fasta contains phos true label
            line = line.strip()
            P_exist=False
            with open('/local2/yuyan/PTM-Motif/Data/Musite_data/fasta/'+line) as fr:
                fa = list(SeqIO.parse(fr, 'fasta'))[0]
            sequence = str(fa.seq)
            uid = str(fa.id)
            if '|' in uid:
                uid = uid.split('|')[1]
            records = cut_protein(sequence, FLAGS.seq_len, 'STY')#label2aa[FLAGS.label] 

            for site in scan[uid]:
                if site[1]!=kin_type:
                    continue
                
        # for k in tqdm(dat):
            # P_exist=False
            # sequence = dat[k]['seq']
            # records = cut_protein(sequence, FLAGS.seq_len, label2aa[ptm_type])
            # label = dat[k]['label']

                preds = {}
                for record in records:
                    seq = record['seq']
                    idx = record['idx']
                    chunk_id = record['chunk_id']
                    if chunk_id is None:
                        name_id = ''
                        chunk_id = 0
                    else:
                        name_id = '~'+str(chunk_id)

                    adj = np.load('./ttt/'+uid+name_id+'_514_5.npy')
                    adj = np.expand_dims(adj, 0)
                    X = pad_X(tokenize_seq(seq), FLAGS.seq_len)
                    X = [tf.expand_dims(X, 0),adj, tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]

                    if chunk_id==0:
                        cover_range = (0,chunk_size//4*3)
                    elif chunk_id==((len(sequence)-1)//-1):
                        cover_range = (chunk_size//4+chunk_id*chunk_size//2, len(sequence))
                    else:
                        cover_range = (chunk_size//4+chunk_id*chunk_size//2, chunk_size//4+(chunk_id+1)*chunk_size//2)

                    seq_idx = site[0]
                    # only get gradient when the seq_idx fall in the range
                    if seq_idx+1 >=cover_range[0] and seq_idx+1 < cover_range[1]:
                        # get gradient for specific ptm
                        seq_idx = seq_idx - chunk_id*chunk_size//2 +1# padding for zero-based
                        # emb_grads = get_gradients(X, emb_model, grad_model, label_to_index[FLAGS.label], seq_idx,method='gradient')
                        fig_name = 'figs/'+'temp'#'_'.join([FLAGS.protein,str(FLAGS.seq_idx), FLAGS.label])
                        # heatmap(emb_grads, seq_idx, fle=(fig_name+'_gradient.png', fig_name+'_local_gradient.png'))
                        
                        pred_score =  model(X).numpy()
                        pred_score = pred_score.reshape(1, -1,13)
                        if pred_score[0, seq_idx+1, label_to_index[ptm_type]] < prob_thres:
                            continue
                        temp_seq = seq[seq_idx-7:seq_idx+8]
                        if len(temp_seq)==15:
                            seqs.append(temp_seq)

                        # get intergrated gradient for specific ptm
                        emb = emb_model(X)
                        m_steps = 50
                        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.
                        # pad_baseline = [tf.expand_dims(['<PAD>']*FLAGS.seq_len, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])]
                        # interpolated_emb, baseline = interpolate_emb(emb, alphas, pad_baseline)
                        interpolated_emb, baseline = interpolate_emb(emb, alphas, seq_idx+1)
                        #TODO try <PAD>
                        if interpolated_emb is None:
                            continue
                        emb_grads= get_gradients(X, emb_model, grad_model, label_to_index[ptm_type], \
                            seq_idx+1, interpolated_emb, method='integrated_gradient', emb=tf.tile(emb,(21,1,1)), baseline=baseline)
                        w_size = 7
                    
                        emb_grads = np.squeeze(emb_grads)
                        left = seq_idx-w_size
                        right = seq_idx+w_size+1
                        # if left <0:
                        #     left = 0
                        # else:
                        #     seq_idx=w_size
                        # if right > len(emb_grads):
                        #     right=len(emb_grads)
                        emb_grads = emb_grads[3:18]
                        all_saliency.append(emb_grads)
                        optm_count+=1
                        P_exist=True
                if P_exist:
                    protein_count+=1
                if len(all_saliency)==0:
                    print('no ptm predicted for '+ ptm_type)
                    continue
        print(optm_count)
        print(protein_count)
        all_saliency = np.stack(all_saliency,axis=0)
        norm_sal = np.divide(all_saliency,np.linalg.norm(all_saliency, axis=1, keepdims=True))
        # norm_sal = np.sum(norm_sal,axis=0)
        heatmap(norm_sal,25, 
            '/local2/yuyan/PTM-Motif/Data/OPTM/pattern/'+kin_type+'_local.pdf')
        # dist_plot(all_saliency, '/local2/yuyan/PTM-Motif/Data/OPTM/pattern/'+ptm_type+'_dist.pdf', thres=3)
        np.save('/local2/yuyan/PTM-Motif/Data/OPTM/pattern/'+kin_type+'.npy', all_saliency) 
        with open('/local2/yuyan/PTM-Motif/Data/OPTM/pattern/'+ptm_type+'_seq.txt','w') as fw:
            for single_seq in seqs:
                fw.write(single_seq+'\n')
        # pdb.set_trace()
        # heatmap(emb_grads, seq_idx, fle=(fig_name+'_integrated_gradient.png',fig_name+'_local_integrated_gradient.png'))
    # pprint(preds) Q3SYY2 342 glyco_N




if __name__ == '__main__':
    app.run(main)
