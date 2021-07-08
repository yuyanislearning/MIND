from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np
import random
import sys
try:
    import ujson as json
except:
    import json
from Bio import SeqIO



def handle_flags():
    flags.DEFINE_string("tflog",
            '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    # Data configuration.
    flags.DEFINE_string('config',
            'config.yml', 'configure file (default: config.yml)')


    # Model parameters.
    flags.DEFINE_bool("multilabel", True, "multilabel or not (default: True)")

    # Training parameters.
    flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
    flags.DEFINE_integer("num_epochs",
            20, "Number of training epochs (default: 20)")
    flags.DEFINE_integer('random_seed',
            252, 'Random seeds for reproducibility (default: 252)')
    flags.DEFINE_float('learning_rate',
            1e-3, 'Learning rate while training (default: 1e-3)')
    flags.DEFINE_float('l2_reg',
            1e-3, 'L2 regularization lambda (default: 1e-3)')
    FLAGS = flags.FLAGS


def limit_gpu_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
          print(e)
          return False
    return True


class Data:
    def __init__(self, file_name, FLAGS):
        self.batch_size = FLAGS.batch_size
        self.records = []

        convert_ptm = {'ac':1, 'ga':2,'gl':3,
                'm1':4,'p':5,'sm':6,'ub':7}

        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(file_name, 'r') as fp:
            dat = json.load(fp)
            for k in dat.keys():
                sequence = str(dat[k]['sequence'])
                labels = dat[k]['labels']
                # for multi label
                if FLAGS.multilabel:
                    feed_label = np.zeros((7, len(sequence))) 
                else:
                    # for single label, PTM or not
                    feed_label = np.zeros((len(sequence)))
                for label in labels:
                    if label['site']-1>len(sequence):
                        continue
                    if FLAGS.multilabel:
                        feed_label[convert_ptm[label['ptm_type']] -1 , label['site']-1] = 1
                    else:
                        feed_label[label['site']-1] = 1
                self.records.append({
                    'uid': k,
                    'seq': sequence,
                    'label':feed_label
                })
        logging.info('Loaded {} records from {}.'.format(len(self.records),
            file_name))

    # def pad(self, x):
    #     y = np.zeros(self.L * self.max_len, dtype=np.int32)
    #     RL = min(len(x), self.L * self.max_len)
    #     y[:RL] = x[:RL]
    #     return y

    def batch_iter(self, is_random=True):
        if is_random:
            random.shuffle(self.records)
        cur_seq, cur_uid,  cur_lbl = [], [], []

        cur_cnt = 0
        for data in self.records:
            cur_lbl.append([data['label']])
            cur_uid.append(self.pad(data['uid']))
            cur_seq.append(self.pad(data['seq']))
            cur_cnt += 1
            if cur_cnt == self.batch_size:
                yield {
                        'label': np.array(cur_lbl),
                        'seq':cur_seq,
                        'label':cur_lbl}
                cur_cnt = 0
                cur_seq, cur_uid,  cur_lbl = [], [], []
        yield {
                'label': np.array(cur_lbl),
                'seq':cur_seq,
                'label':cur_lbl}

