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

import pdb

from src.utils import get_class_weights,  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import GAT_model,  RNN_model, TransFormer
from src.tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token
from src.transformer import  positional_encoding


handle_flags()
def main(argv):

    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    limit_gpu_memory_growth()

    file_name = '/workspace/PTM/Data/Musite_data/ptm/PTM_test.json'
    with open(file_name, 'r') as fp:
        # data structure: {PID:{seq, label:[{site, ptm_type}]}}
        dat = json.load(fp)

    seq = dat['Q02388']['seq']
    labels = dat['Q02388']['label']
    seq_len = len(seq) + 3

    X = pad_X(tokenize_seqs([seq]), seq_len)
    X = [X, tf.tile(positional_encoding(seq_len,128), [1,1,1])]

    model = tf.keras.models.load_model('saved_model/Transformer/Transformer__multi_514/')

    y_pred = model.predict(X)
    y_pred = y_pred.reshape(1, seq_len, -1)
    y_pred = y_pred[:,:,0]
    y_pred = y_pred>=0.5

    all_index = [i for i,s in enumerate(seq) if s=='K']
    y_trues = {i:0 for i in all_index}
    for l in labels:
        if l['ptm_type']=='Hydro_K':
            y_trues[l['site']] = 1
    
    total = {i:(y_pred[0,i+1],y_trues[i]) for i in all_index}
    pprint(total)

    pdb.set_trace()


def pad_X( X, seq_len):
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in X])
    

def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]



if __name__ == '__main__':
    app.run(main)
