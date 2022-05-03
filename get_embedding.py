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
import pandas as pd
import re
from sklearn.manifold import TSNE


import pdb

from src.utils import get_class_weights,  limit_gpu_memory_growth, PTMDataGenerator
from src.model import TransFormerFixEmbed,  RNN_model, TransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding


def handle_flags():
    flags.DEFINE_string('model_path',
            'saved_model/LSTMTransformer/LSTMTransformer_514_multin_layer_3_fold_0', 'pretrained model path to load ')
    flags.DEFINE_string('data_path', './', 'path to fasta data')#268 Ubi_K
    flags.DEFINE_string('res_path','analysis/figures', 'path to result dir')
    flags.DEFINE_string('adj_path', '', 'path to structure adjency matrix')
    flags.DEFINE_string('model', 'Transformer', 'path to structure adjency matrix')


    # Model parameters.
    flags.DEFINE_bool("graph", False, "use only spatially and locally close sequence (default: False)")#TODO
    flags.DEFINE_bool("ensemble", False, "ensemble learning")#TODO
    flags.DEFINE_bool("random_ensemble", False, "ensemble learning")#TODO
    flags.DEFINE_bool("neg_sam", False, "ensemble learning")#TODO
    flags.DEFINE_bool("dat_aug", False, "ensemble learning")#TODO
    flags.DEFINE_bool("embedding", False, "ensemble learning")#TODO

    # Training parameters.
    flags.DEFINE_integer("seq_len", 514, "maximum lenth+2 of the model sequence (default: 512)")
    flags.DEFINE_integer("d_model", 128, "hidden dimension of the model")
    flags.DEFINE_integer("batch_size", 128, "hidden dimension of the model")
    flags.DEFINE_integer("fill_cont", None, "hidden dimension of the model")


    FLAGS = flags.FLAGS

def tsne_plot(embs, true_labels, fig_path):
    X_embedded = TSNE().fit_transform(embs)
    dat = pd.DataFrame(data = {'X':X_embedded[:,0], 'Y':X_embedded[:,1], 'ptm':true_labels})
    sns.scatterplot(data=dat, x='X', y='Y', hue='ptm', s=10, dpi=300)
    plt.show()
    plt.savefig(fig_path, dpi=300)
    plt.close()


handle_flags()
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

    model = tf.keras.models.load_model(FLAGS.model_path)
    
    input_emb_model = keras.models.Model(
        [model.inputs], [model.get_layer('embedding').output]
    )
    
    output_emb_model = keras.models.Model(
        [model.inputs], [model.get_layer('encoder_layer_2').output]
    )
    chunk_size = FLAGS.seq_len - 2

    
    test_dat_aug = PTMDataGenerator('/workspace/PTM/Data/Musite_data/ptm/PTM_test.json', FLAGS, shuffle=True,ind=None, eval=True)

    y_in_embs = []
    y_out_embs = []
    y_true_labels = []
    for test_X,test_Y,test_sample_weights in test_dat_aug:
        batch = test_Y.shape[0]
        y_in_emb = input_emb_model.predict(test_X, batch_size=FLAGS.batch_size).reshape(batch*FLAGS.seq_len, -1) #(batch*seq_len,dim)
        y_out_emb = output_emb_model.predict(test_X, batch_size=FLAGS.batch_size)[0][0].reshape(batch*FLAGS.seq_len, -1) #(batch*seq_len,dim)
        y_true = test_Y.reshape( -1, len(unique_labels))# (batch* seq_len, 13)
        y_true_exist = tf.math.reduce_sum(y_true, axis=-1)>0
        y_true_label_index = tf.math.argmax(y_true, -1)
        y_in_embs.append( y_in_emb[y_true_exist])
        y_out_embs.append( y_out_emb[y_true_exist])
        y_true_labels.append(y_true_label_index[y_true_exist])
        
    y_in_embs = tf.concat(y_in_embs, 0)
    y_out_embs = tf.concat(y_out_embs, 0)
    y_true_labels = tf.concat(y_true_labels, 0)
    y_true_labels = [index_to_label[ll] for ll in y_true_labels.numpy()]
    tsne_plot(y_in_embs, y_true_labels, FLAGS.res_path+'/tsne_input_embedding.png')
    tsne_plot(y_out_embs, y_true_labels, FLAGS.res_path+'/tsne_output_embedding.png')


if __name__ == '__main__':
    app.run(main)
