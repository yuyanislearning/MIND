#!/usr/bin/env python3
from absl import app, flags
from absl import logging
import random
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras

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

from src.utils import get_class_weights, get_unique_labels, Data, handle_flags, limit_gpu_memory_growth
from src import utils
from src.model import RNN_model

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

    # Load data
    cfg = yaml.load(open(FLAGS.config, 'r'), Loader=yaml.BaseLoader) #TODO
    data_prefix = '{}/PTM_'.format(
            cfg['path_data']) 
    path_pred  = '{}/PTM_'.format(
            cfg['path_pred'])

    train_data = utils.Data(data_prefix + 'train.json', FLAGS)
    test_data = utils.Data(data_prefix + 'test.json', FLAGS)
    val_data = utils.Data(data_prefix+'val.json', FLAGS)

    unique_labels = get_unique_labels(train_data, val_data, test_data)
    class_weights = get_class_weights(train_data, val_data, test_data, unique_labels) if FLAGS.class_weights else None
    train_data.encode_data( FLAGS.seq_len,  unique_labels, class_weights)
    test_data.encode_data( FLAGS.seq_len,  unique_labels, negative_sampling=False)
    val_data.encode_data( FLAGS.seq_len,  unique_labels,  negative_sampling=False)

    optimizer = tf.keras.optimizers.Adam(
            learning_rate=FLAGS.learning_rate, amsgrad=True)

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # Build model
    model = RNN_model(optimizer, loss_object)
    model.create_model(FLAGS.seq_len, 128, unique_labels)

    # Optimization settings.

    model.train( train_data, val_data, FLAGS.seq_len, FLAGS.batch_size, FLAGS.num_epochs, lr = None, callbacks=training_callbacks)

    logging.info('------------------evaluate---------------------' )

    model.eval(FLAGS.seq_len,test_data, FLAGS.batch_size, unique_labels)


    

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

