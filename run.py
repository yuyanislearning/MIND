#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
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

import utils

utils.handle_flags()

def main(argv):
    FLAGS = flags.FLAGS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tflog

    utils.limit_gpu_memory_growth()
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)
    tf.random.set_seed(FLAGS.random_seed)

    cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader) #TODO
    data_prefix = '{}/PTM_{}'.format(
            cfg['path_data']) 
    path_pred  = '{}/PTM_{}'.format(
            cfg['path_pred'])

    train_data = utils.Data(data_prefix + 'train.json', FLAGS)
    test_data = utils.Data(data_prefix + 'test.json', FLAGS)
    val_data = utils.Data(data_prefix+'valid.json', FLAGS)

    model = PTM_pattern_finder#TODO: add the model

    # Optimization settings.
    loss_object = tf.keras.losses.BinaryCrossentropy()

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    # Logging metric settings.
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # TF Functions.
    @tf.function
    def train_step(data):
        with tf.GradientTape() as tape:
            predictions = model(
                    data['label'],
                    data['seq'],)
            loss = loss_object(data['label'], predictions)
            loss += sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        return predictions 
    
    @tf.function
    def valid_step(data, loss_metric):
        predictions = model(
                data['label'],
                data['seq'])
        loss = loss_object(data['label'], predictions)
        
        loss_metric(loss)
        return predictions 
    
    def eval(y_true, y_pred, multilabel):
        if multilabel:
            #TODO
        else:
            y_true = [1 if x > 0.5 else -1 for x in y_true]
            y_pred = [1 if x > 0.5 else -1 for x in y_pred]
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred, pos_label=1)
            f1  = f1_score(y_true, y_pred)
            mcc = matthews_corrcoef(y_true, y_pred)
            sen = recall_score(y_true, y_pred, pos_label=1)
            spe = recall_score(y_true, y_pred, pos_label=-1)        
            return acc, pre, f1, mcc, sen, spe

    # Training and Evaluating.
    best_f1 = 0.0
    for epoch in range(FLAGS.num_epochs):
        # Reset metrics.
        train_loss.reset_states()
        # Training.
        num_batches = (len(train_data.records) + FLAGS.batch_size - 1)
        num_batches = num_batches // FLAGS.batch_size
        preds, lbls = [], []
        for data in tqdm(train_data.batch_iter(), desc='Training',
                total=num_batches):
            preds.extend(list(train_step(data)))
            lbls.extend(list(data['label']))
        train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe = \
                eval(lbls, preds)

        tmpl = 'Epoch {} (CV={}, K={}, L={})\n' +\
                'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
        print(tmpl.format(
            epoch + 1, FLAGS.cv, FLAGS.K, FLAGS.L,
            train_loss.result(),
            train_acc, train_pre, train_f1, train_mcc, train_sen, train_spe),
            file=sys.stderr)

    # Testing and Evaluating.
    # Reset metrics.
    test_loss.reset_states()
    # Training.
    num_batches = (len(test_data.records) + FLAGS.batch_size - 1)
    num_batches = num_batches // FLAGS.batch_size
    preds, lbls = [], []
    for data in tqdm(test_data.batch_iter(is_random=False),
            desc='Testing', total=num_batches):
        preds.extend(list(valid_step(data, test_loss)))
        lbls.extend(list(data['label']))

    lbls = [int(x) for x in lbls]
    preds = [float(x) for x in preds]
    test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe = \
            eval(lbls, preds)

    tmpl = 'Testing (CV={}, K={}, L={})\n' +\
            'Ls: {}\tA: {}\t P: {}\tF: {},\tM: {}\tSe: {}\tSp: {}\n'
    print(tmpl.format(FLAGS.cv, FLAGS.K, FLAGS.L,
        test_loss.result(),
        test_acc, test_pre, test_f1, test_mcc, test_sen, test_spe),
        file=sys.stderr)

    logging.info('Saving testing predictions to to {}.'.format(path_pred))
    with open(path_pred, 'w') as wp:
        json.dump(list(zip(preds, lbls)), wp)


if __name__ == '__main__':
    app.run(main)

