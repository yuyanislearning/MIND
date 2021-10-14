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

import pdb

try:
    import ujson as json
except:
    import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm
import yaml

from src.utils import get_class_weights, get_unique_labels, Data, handle_flags, limit_gpu_memory_growth, PTMDataGenerator,CategoricalTruePositives
from src import utils
from src.model import ProteinBert, RNN_model

handle_flags()



def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience = 1, factor = 0.25, min_lr = 1e-05, verbose = 1),
        keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = True),
    ]
    tf.config.run_functions_eagerly(True)#TODO remove


    # Load data
    cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader) #TODO
    data_prefix = '{}/PTM_'.format(
            cfg['path_data']) 
    path_pred  = '{}/PTM_'.format(
            cfg['path_pred']) 

    train_data = utils.Data(data_prefix + 'train.json', FLAGS)
    test_data = utils.Data(data_prefix + 'test.json', FLAGS)
    val_data = utils.Data(data_prefix+'val.json', FLAGS)

    # setting up
    unique_labels = get_unique_labels(train_data, val_data, test_data)

    class_weights = get_class_weights(train_data, val_data, test_data, unique_labels) if FLAGS.class_weights else None
    # if FLAGS.short:# only use partial sequence
    #     train_data.encode_data_short( FLAGS.seq_len,  unique_labels, is_binary=FLAGS.binary,  spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert')
    #     test_data.encode_data_short( FLAGS.seq_len,  unique_labels, is_binary=FLAGS.binary,  spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert')
    #     val_data.encode_data_short( FLAGS.seq_len,  unique_labels, is_binary=FLAGS.binary,  spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert')
    # elif FLAGS.graph:# encode data for graph
    #     train_data.encode_data_graph( FLAGS.seq_len,  unique_labels, class_weights, is_multilabel=FLAGS.multilabel, \
    #         is_binary=FLAGS.binary, spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=False, train_val_test='train')
    #     test_data.encode_data_graph( FLAGS.seq_len,  unique_labels, is_multilabel=FLAGS.multilabel, \
    #         is_binary=FLAGS.binary,negative_sampling=False,spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=True, train_val_test='test')
    #     val_data.encode_data_graph( FLAGS.seq_len,  unique_labels, is_multilabel=FLAGS.multilabel, \
    #         is_binary=FLAGS.binary, negative_sampling=False, spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=True, train_val_test='val')
    # else:
    train_data.encode_data( FLAGS.seq_len,  unique_labels, class_weights, is_multilabel=FLAGS.multilabel, graph=FLAGS.graph,\
        is_binary=FLAGS.binary, spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=False, train_val_test='train')
    test_data.encode_data( FLAGS.seq_len,  unique_labels, is_multilabel=FLAGS.multilabel,  graph=FLAGS.graph,\
        is_binary=FLAGS.binary,negative_sampling=False,spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=True, train_val_test='test')
    val_data.encode_data( FLAGS.seq_len,  unique_labels, is_multilabel=FLAGS.multilabel,  graph=FLAGS.graph,\
        is_binary=FLAGS.binary, negative_sampling=False, spec_neg_sam=FLAGS.spec_neg_sam, proteinbert=FLAGS.model=='proteinbert', evaluate=True, train_val_test='val')

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)



    # metrics = [CategoricalTruePositives(13,batch_size=FLAGS.batch_size)]#,tf.keras.metrics.FalsePositives(),tf.keras.metrics.TrueNegatives(),tf.keras.metrics.TruePositives()
    metrics = []
    # Build model
    if FLAGS.model=='proteinbert':
        model = ProteinBert(optimizer, loss_object, unique_labels, FLAGS.learning_rate,FLAGS.binary)
        model.create_model( train_data,  FLAGS.seq_len,   binary=FLAGS.binary, freeze_pretrained_layers=False, graph=FLAGS.graph)
    else:
        model = RNN_model(optimizer, loss_object, FLAGS.learning_rate)
        model.create_model(FLAGS.seq_len, 128, unique_labels, 0.6,metrics, FLAGS.binary, FLAGS.multilabel, FLAGS.graph)

    # Optimization settings.
    if not FLAGS.binary:# multi-label
        
        model.train( train_data, val_data, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, unique_labels, lr = FLAGS.learning_rate, callbacks=training_callbacks,graph=FLAGS.graph)
        logging.info('------------------evaluate---------------------' )

        AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_data, FLAGS.batch_size, unique_labels, FLAGS.graph)
        for u in unique_labels:
            # print(u)
            print('%.3f'%PR_AUC[u])
        for u in unique_labels:
            print(u)
            print(confusion_matrixs[u])


    if FLAGS.binary:
        # train on large samples first
        sort_ind = np.argsort(-np.array([train_data.Y[i].shape[0] for i in range(len(train_data.Y))]))
        AUCs, PR_AUCs, confu_mats = {}, {}, {}
        for i in sort_ind:
            print('training on '+ unique_labels[i])
            model.train(train_data, val_data, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, lr = FLAGS.learning_rate, callbacks=training_callbacks,metrics=metrics, binary=FLAGS.binary, ind=i)
            AUC, PR_AUC, confusion_matrixs = model.eval(FLAGS.seq_len,test_data, FLAGS.batch_size, unique_labels, binary=FLAGS.binary, ind=i)
            AUCs[unique_labels[i]], PR_AUCs[unique_labels[i]], confu_mats[unique_labels[i]] = AUC, PR_AUC, confusion_matrixs

        for i in sort_ind:
            u = unique_labels[i]
            print(u)
            #print('%.3f'%AUCs[u])
            print('%.3f'%PR_AUCs[u])
        for i in sort_ind:
            u = unique_labels[i]
            print(u)    
            print(confu_mats[u])
            

    

    # # Training and Evaluating.
    # best_f1 = 0.0
    # for epoch in range(FLAGS.num_epochs):
    #     # Reset metrics.
    #     train_loss.reset_states()
    #     # Training.
    #     num_batches = (len(train_data.records) + FLAGS.batch_size - 1)
    #     num_batches = num_batches // FLAGS.batch_size
    #     preds, lbls = [], []
    #     for data in tqdm(train_data.batch_iter(), desc='Training',
    #             total=num_batches):
    #         preds.extend(list(train_step(data)))
    #         lbls.extend(list(data['label']))
    #     train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe = \
    #             eval(lbls, preds)

    #     tmpl = 'Epoch {} (CV={}, K={}, L={})\n' +\
    #             'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    #     print(tmpl.format(
    #         epoch + 1, FLAGS.cv, FLAGS.K, FLAGS.L,
    #         train_loss.result(),
    #         train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe),
    #         file=sys.stderr)

    # # Testing and Evaluating.
    # # Reset metrics.
    # test_loss.reset_states()
    # # Training.
    # num_batches = (len(test_data.records) + FLAGS.batch_size - 1)
    # num_batches = num_batches // FLAGS.batch_size
    # preds, lbls = [], []
    # for data in tqdm(test_data.batch_iter(is_random=False),
    #         desc='Testing', total=num_batches):
    #     preds.extend(list(valid_step(data, test_loss)))
    #     lbls.extend(list(data['label']))

    # lbls = [int(x) for x in lbls]
    # preds = [float(x) for x in preds]
    # test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe = \
    #         eval(lbls, preds)

    # tmpl = 'Testing (CV={}, K={}, L={})\n' +\
    #         'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    # print(tmpl.format(FLAGS.cv, FLAGS.K, FLAGS.L,
    #     test_loss.result(),
    #     test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe),
    #     file=sys.stderr)

    # logging.info('Saving testing predictions to to {}.'.format(path_pred))
    # with open(path_pred, 'w') as wp:
    #     json.dump(list(zip(preds, lbls)), wp)




if __name__ == '__main__':
    app.run(main)

