from absl import flags
from absl import logging
import tensorflow as tf
import numpy as np
import random
import sys
import pdb
try:
    import ujson as json
except:
    import json
from Bio import SeqIO

from .tokenization import additional_token_to_index, n_tokens, tokenize_seq




def handle_flags():
    flags.DEFINE_string("tflog",
            '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    # Data configuration.
    flags.DEFINE_string('config',
            'config.yml', 'configure file (default: config.yml)')


    # Model parameters.
    flags.DEFINE_bool("multilabel", False, "multilabel or not (default: True)")
    flags.DEFINE_bool("binary", True, "Binary or not (default: True)")
    flags.DEFINE_bool("spec_neg_sam", True, "use ptm-specific negative sampling or not (default: True)")
    flags.DEFINE_bool("class_weights", False, "use class weights or not (default: True)")

    # Training parameters.
    flags.DEFINE_integer("seq_len", 512, "maximum lenth+2 of the model sequence (default: 512)")
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
        self.records = []
        # ptm_list = ['Phos_ST','Phos_Y','glyco_N','glyco_ST','Ubi_K','SUMO_K','N6-ace_K','Methy_R','Methy_K','Pyro_Q','Palm_C','Hydro_P','Hydro_K']
        # convert_ptm = {p:i+1 for i, p in enumerate(ptm_list)}
        flatten = lambda l: [item for sublist in l for item in sublist]
        with open(file_name, 'r') as fp:
            dat = json.load(fp)
            for k in dat:
                # some case that the data miss sequence, skip
                if dat[k].get('seq',-1)==-1:
                    continue
                sequence = str(dat[k]['seq'])
                labels = dat[k]['label']
                self.records.append({
                    'uid': k,
                    'seq': sequence,
                    'label':labels
                })
        logging.info('Loaded {} records from {}.'.format(len(self.records),
            file_name))
    def encode_data( self, seq_len,  unique_labels, class_weights=None, negative_sampling=False,is_binary=True, is_multilabel=False, spec_neg_sam=True):
        data_seq = [d['seq'] for d in self.records]
        data_label = [ d['label'] for d in self.records]
        data_uid = [d['uid'] for d in self.records]        
        self.filter_records = [self.records[i] for i, r in enumerate(self.records) if len(r['seq'])<=(seq_len-2)]
        
        data_seq = [d['seq'] for d in self.filter_records]
        data_label = [ d['label'] for d in self.filter_records]
        data_uid = [d['uid'] for d in self.filter_records]
        if is_multilabel:
            label_to_index = {str(label): i for i, label in enumerate(unique_labels)}
            Y = np.zeros((len(data_seq), seq_len, len(unique_labels))) 
            sample_weights = np.zeros((len(data_seq), seq_len, len(unique_labels)))

            for i, seq in enumerate(data_seq):
                pos_ind = []
                for j, lbl in enumerate(data_label[i]):
                    #pdb.set_trace()
                    # for random case that site greater than seq
                    if int(lbl['site']) > len(data_seq):
                        continue
                    Y[i, int(lbl['site'])+1, label_to_index[lbl['ptm_type']]] = 1
                    sample_weights[i, int(lbl['site'])+1, label_to_index[lbl['ptm_type']]] =1
                    pos_ind.append((int(lbl['site']),label_to_index[lbl['ptm_type']]))
                if negative_sampling:
                    sam_ind = np.array(choices([i for i in range(len(data_seq))],  k = len(pos_ind)*10)).astype(int)
                    lbl_ind = np.array(choices([i for i in range(len(unique_labels))],  k=len(pos_ind)*10)).astype(int)
                    count = 0
                    for sa, lb in zip(sam_ind, lbl_ind):
                        if (sa,lb) in pos_ind:
                            continue
                        sample_weights[i, sa+1, lb] = 1
                        count+=1
                        if count >= len(pos_ind):
                            break
                else:
                    sample_weights[i, 1:(1+len(data_seq)), :] = 1
            
            Y = Y.reshape((len(data_seq),-1, 1))
            if class_weights is None:
                class_weights = np.repeat([1], len(unique_labels))
            sample_weights = np.tile(class_weights, (len(data_seq), seq_len, 1)) * sample_weights
            sample_weights = sample_weights.reshape((len(data_seq), -1, 1))
        if is_binary:
            all_Y = []
            all_wei = []
            all_X = []

            for ul in unique_labels:
                Y = np.zeros((len(data_seq), seq_len)) 
                sample_weights = np.zeros((len(data_seq), seq_len))
                for i, seq in enumerate(data_seq):
                    ptm_type = set()
                    pos_ind = []
                    for j, lbl in enumerate(data_label[i]):
                        if lbl['ptm_type']==ul:
                            Y[i, int(lbl['site'])+1] = 1
                            sample_weights[i, int(lbl['site'])+1] =1
                            ptm_type = ptm_type.union(seq[lbl['site']])
                            pos_ind.append(lbl['site'])
                    if spec_neg_sam:
                        if len(ptm_type)>0:
                            neg_ind = set([ii for ii, aa in enumerate(seq) if aa in list(ptm_type)])
                            neg_ind = np.array(list(neg_ind - set(pos_ind))).astype(int)
                            if len(neg_ind)>0:
                                k = min(len(pos_ind), len(neg_ind))
                                neg_ind = np.random.choice(neg_ind, k, False)
                                sample_weights[i, neg_ind+1] = 1
                pos_y_ind = np.where(np.sum(Y, axis = 1)>0)[0]
                all_Y.append(np.expand_dims(Y[pos_y_ind,:], axis = 2))
                all_wei.append(np.expand_dims(sample_weights[pos_y_ind,:], axis = 2))
                temp_data_seq = [data_seq[ii] for ii in pos_y_ind]
                all_X.append(tokenize_seqs(temp_data_seq, seq_len))
            Y = all_Y
            sample_weights = all_wei
            X = all_X   
                   
        self.Y  = Y
        self.sample_weights = sample_weights
        
        # Encode X
        if not is_binary:
            X = tokenize_seqs(data_seq, seq_len)
        self.X = X
            
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

def get_unique_labels(train_set, valid_set, test_set):
    return sorted( set([l['ptm_type'] for d in train_set.records for l in d['label'] ]).union(\
                set([l['ptm_type'] for d in valid_set.records for l in d['label'] ])).union(\
                    set([l['ptm_type'] for d in test_set.records for l in d['label'] ])))


def get_class_weights(train_set, valid_set, test_set, unique_labels):
    class_weights = {u:0 for u in unique_labels}

    for d in train_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    for d in valid_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    for d in test_set.records:
        for l in d['label']:
            class_weights[l['ptm_type']]+=1 
    
    class_weights = [float(class_weights[u]) for u in unique_labels]
    mean_w = sum(class_weights)/len(class_weights)
    class_weights = np.array([mean_w/u for u in class_weights])
    return class_weights

def tokenize_seqs(seqs, seq_len):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in map(tokenize_seq, seqs)], dtype = np.int32)
