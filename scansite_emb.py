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


import pdb
from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator, handle_flags
from src import utils
from src.model import TransFormerFixEmbed,  RNN_model, LSTMTransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding
# get saliency for scansite motif prediction and check clustering

model_name = 'saved_model/con_g'
fold = 1
prob_thres = 0.5
handle_flags()
ensemble=False


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

    with open('/local2/yuyan/PTM-Motif/Data/Musite_data/Phosphorylation_motif/correct_scan_kinase.json') as f:
        scan = json.load(f)

    fp =  open('/local2/yuyan/PTM-Motif/Data/Musite_data/Phosphorylation_motif/all_phos_uid.txt', 'r')

    if ensemble:
        with open(model_name+'_PRAU.json') as f:
            AUPR_dat = json.load(f)
    optimizer = tf.keras.optimizers.Adam(
                        learning_rate=FLAGS.learning_rate, amsgrad=True)


    models = [] # load models
    grad_models = []
    emb_models = []
    for i in range(fold):
        if ensemble:
            model = tf.keras.models.load_model(model_name+'_fold_'+str(i))
        else:
            model = tf.keras.models.load_model(model_name)
        models.append(model)
        grad_model = build_model_graph(FLAGS, optimizer , unique_labels, model)
        grad_models.append(grad_model.model)
        emb_model = keras.models.Model(
            [model.inputs], [model.get_layer('embedding').output]
        )
        emb_models.append(emb_model)
        

    # emb_model = keras.models.Model(
    #     [models[0].inputs], [models[0].get_layer('encoder_layer').output]
    # )
    if ensemble:
        weights = ensemble_get_weights(AUPR_dat, unique_labels)

    embs = []
    kinases = []
    seqs = []
    # get all the saliency scores
    for line in tqdm(fp): # for every fasta contains phos true label
        line = line.strip()
        with open('/local2/yuyan/PTM-Motif/Data/Musite_data/fasta/'+line) as fr:
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

            for site in scan[uid]:
                if site[0] >=cover_range[0] and site[0]< cover_range[1]:  
                    seq_idx = site[0] - chunk_id*chunk_size//2 
                    # get intergrated gradient for specific ptm
                    emb_grads = []
                    probs = []
                    if ensemble:
                        for i in range(fold):
                            emb = emb_models[i](X)
                            m_steps = 50
                            alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1) # Generate m_steps intervals for integral_approximation() below.

                            # pad_seq = [additional_token_to_index['<START>']] + [additional_token_to_index['<PAD>']]*(FLAGS.seq_len-2) +[additional_token_to_index['<END>']]
                            # pad_baseline = emb_model([tf.expand_dims(pad_seq, 0), tf.tile(positional_encoding(FLAGS.seq_len, FLAGS.d_model), [1,1,1])])
                            # interpolated_emb, baseline = interpolate_emb(emb, alphas, seq_idx, 'pad',pad_baseline)
                            interpolated_emb, baseline = interpolate_emb(emb, alphas, seq_idx)# TODO seq_idx +1
                            if baseline is None:
                                continue
                            temp_label = 'Phos_ST' if seq[seq_idx] in 'ST' else 'Phos_Y'
                            emb_grad, prob = get_gradients(X, emb_models[i], grad_models[i], label_to_index[temp_label], \
                                seq_idx, interpolated_emb, method='integrated_gradient', emb=tf.tile(emb,(21,1,1)), baseline=baseline, graph = adj)
                            prob = prob.numpy()
                            if ensemble:
                                probs.append(weights[temp_label][i]*prob)
                                emb_grads.append(weights[temp_label][i]*emb_grad)
                            else:
                                emb_grads = emb_grad
                        if ensemble:
                            prob = np.sum(np.array(probs))
                            emb_grads# TODO
                    else:
                        temp_label = 'Phos_ST' if seq[seq_idx] in 'ST' else 'Phos_Y'
                        pred_score =  models[0](X).numpy()
                        pred_score = pred_score.reshape(1, -1,13)
                        if pred_score[0, seq_idx+1, label_to_index[temp_label]] < prob_thres:
                            continue
                        emb = emb_models[0](X)
                        m_steps = 50
                        alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps+1)
                        interpolated_emb, baseline = interpolate_emb(emb, alphas, seq_idx+1)
                        if baseline is None:
                            continue
                        emb_grad = get_gradients(X, emb_models[i], grad_models[i], label_to_index[temp_label], \
                            seq_idx+1, interpolated_emb, method='integrated_gradient', emb=tf.tile(emb,(21,1,1)), baseline=baseline, graph = adj)
                        # prob = prob.numpy()
                        emb_grads = emb_grad

                    embs.append(emb_grad)
                    kinases.append(site[1])
                    seqs.append(seq[seq_idx-10:seq_idx+11])

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
    embs = tf.stack(embs, 0).numpy()
    kinases = np.array(kinases)
    seqs = np.array(seqs)
    # for kina in all_st_kinase:
    #     select_index = [k for k,kin in enumerate(kinases) if kin in kina]
    #     select_embs = embs[select_index,]
    #     select_kinases = kinases[select_index,]
    #     dist_plot(select_embs, 'analysis/figures/kinase_dist/'+kina+'.png', thres=3)
    #     dist_plot2(select_embs, 'analysis/figures/kinase_dist/sum_'+kina+'.png')
        # good shape: ATM_kin

    np.save('./saliency/dat/temp_'+str(prob_thres)+'.npy', embs)
    np.save('./saliency/dat/temp_kin_'+str(prob_thres)+'.npy', kinases)
    np.save('./saliency/dat/temp_seq_'+str(prob_thres)+'.npy', seqs)
    # select_index = [k for k,kin in enumerate(kinases) if kin in all_st_kinase]
    # select_embs = embs[select_index,]
    # select_kinases = kinases[select_index,]
    
    # group_kinases = []
    # for kin in select_kinases:
    #     if kin in acid_kinase:
    #         group_kinases.append('Acidophilic')
    #     elif kin in base_kinase:
    #         group_kinases.append('Basophilic')
    #     elif kin in DNA_damage_kinase:
    #         group_kinases.append('DNA damage')
    #     elif kin in pro_depend_kinase:
    #         group_kinases.append('Proline-dependent')
    

    # norm_embs = np.divide(embs,np.linalg.norm(embs, axis=1, keepdims=True))
    # smooth_embs = [smooth(emb,window_len=11,window='hanning') for emb in embs]
    # smooth_embs = np.stack(smooth_embs)
    # thres_embs = [thres(emb) for emb in embs]
    # thres_embs = np.stack(thres_embs)
    # tsne_plot(norm_embs, kinases, 'analysis/figures/kinase_emb_tsne.png')
    # pca_plot(norm_embs, kinases, 'analysis/figures/kinase_emb_pca.png')
    # tsne_plot(thres_embs, kinases, 'analysis/figures/kinase_emb_tsne.png',perplexity=5)
    # pca_plot(thres_embs, kinases, 'analysis/figures/kinase_emb_pca.png')


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

def dist_plot2(embs, fig_path):
    embs = np.sum(embs, axis=0)
    fig, ax = plt.subplots(figsize=(10,5), layout='constrained')
    embs = np.squeeze(embs)
    ax.plot(list(range(len(embs))), embs)
    ax.scatter(10, embs[10], 50, facecolors='none', edgecolors='black', linewidths=1.5)
    # ax = sns.heatmap(a)
    
    # sns.lineplot(list(range(len(a))), a)
    # plt.plot(highlight_idx, a[highlight_idx], markersize=29, fillstyle='none', markeredgewidth=1.5)
    plt.show()
    plt.savefig(fig_path)

def tsne_plot(embs, true_labels, fig_path, perplexity=30):
    X_embedded = TSNE(perplexity=perplexity).fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'kinase':true_labels})
    sns.scatterplot(data=dat, x='X', y='Y', hue='kinase', s=10, palette='Set1')
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    

def pca_plot(embs, kinases, fig_path):
    pca = PCA(n_components=2)
    X_embedded = pca.fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'kinase':kinases})
    sns.scatterplot(data=dat, x='X', y='Y', hue='kinase', s=10, palette='Set1')
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()    

def thres(emb, thres=5):
    # retain top thres 
    new_emb = np.copy(emb)
    zero_out = thres - emb.shape[0]
    new_emb[np.argpartition(-new_emb,zero_out)[zero_out:]]=0
    new_emb[10] = 0 # remove center
    return new_emb

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also: 
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError
    if x.size < window_len:
        raise ValueError
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

# def create_baseline(seq_len):
#     # create an all padding sequence
#     return np.array(seq_len * [additional_token_to_index['<PAD>']])

def pad_X( X, seq_len):
    return np.array(X + (seq_len - len(X)) * [additional_token_to_index['<PAD>']])

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]




if __name__ == '__main__':
    app.run(main)
