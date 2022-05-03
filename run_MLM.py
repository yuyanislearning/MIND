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
import time

import pdb

try:
    import ujson as json
except:
    import json
from tqdm import tqdm
import yaml

from src.utils import MLMDataGenerator,MLMUniparcDataGenerator,  handle_flags, limit_gpu_memory_growth, PTMDataGenerator
from src import utils
from src.model import GAT_model,  RNN_model, TransFormer, MLMTransFormer

import gc

t0 = time.time()
handle_flags()

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()

def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)
    #tf.config.run_functions_eagerly(True)#TODO remove

    # Load data
    # cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader) #TODO
    if FLAGS.pretrain_name=='PTM':
        data_prefix = '{}/all.json'.format(FLAGS.data_path) 
        prop = 1
        train_dat_aug = MLMDataGenerator(data_prefix, FLAGS, prop , shuffle=True, mask_prop=0.15 )
    if FLAGS.pretrain_name=='Uniparc':
        data_prefix = '/workspace/PTM/Data/MLM_data/uniparc_active.fasta'
        train_dat_aug = MLMUniparcDataGenerator(data_prefix, FLAGS, shuffle=True, mask_prop=0.15 )

    unique_labels = train_dat_aug.unique_labels

    # setting up
    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    # loss_object = loss = keras.losses.CategoricalCrossentropy(from_logits=True, reduction=losses_utils.ReductionV2.NONE)

    model = MLMTransFormer(FLAGS.model, optimizer, d_model=128, \
        num_layers=FLAGS.n_lstm, seq_len=FLAGS.seq_len, num_heads=8,dff=512, rate=0.2,binary=False,\
        unique_labels=unique_labels, split_head=FLAGS.split_head, global_heads=FLAGS.global_heads, fill_cont=FLAGS.fill_cont)
    model.create_model(FLAGS.seq_len, graph=FLAGS.graph)    # Optimization settings.
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    checkpoint_path = './saved_model/MLM_Transformer_{epoch}-{loss:.2f}'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,  verbose=1, save_freq=5000)
    training_callbacks = [
        #keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        # keras.callbacks.EarlyStopping(monitor='loss',patience = 5, restore_best_weights = True),
        cp_callback,
        MyCustomCallback()
    ] 

    # for xx,yy,ss in train_dat_aug:
    #     pdb.set_trace()

    model.model.fit(train_dat_aug,  epochs = FLAGS.num_epochs, steps_per_epoch=train_dat_aug.__len__(), callbacks = training_callbacks)#, validation_steps=val_aug.__len__())
    if FLAGS.save_model:
        model_name = './saved_model/MLM_'+FLAGS.model+'/'+FLAGS.model+'_multi_'+str(FLAGS.seq_len) + '_'+FLAGS.pretrain_name+'_' + str(prop)
        model.model.save(model_name)
    os.system('rm -r '+checkpoint_path)
    
    t1 = time.time()
    total_time = t1-t0
    print('the total time used is:')
    print(total_time)


if __name__ == '__main__':
    app.run(main)

