from absl import flags
from absl import logging
import tensorflow as tf
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, accuracy_score, confusion_matrix

from src.transformer import  positional_encoding

import numpy as np
import random
import sys
import pdb
import math
try:
    import ujson as json
except:
    import json
from Bio import SeqIO
from os.path import exists
from scipy import sparse
from tqdm import tqdm

from .tokenization import additional_token_to_index, n_tokens, tokenize_seq, parse_seq, aa_to_token_index, index_to_token



def handle_flags():
    flags.DEFINE_string("tflog",
            '3', "The setting for TF_CPP_MIN_LOG_LEVEL (default: 3)")
    flags.DEFINE_string('model',
            'proteinbert', 'model to use (default: proteinbert)')
    flags.DEFINE_string('dataset',
            'AF', 'dataset to use (default: Alpha fold pdb available ptms)')
    # Data configuration.
    flags.DEFINE_string('config',
            'config.yml', 'configure file (default: config.yml)')
    flags.DEFINE_string('data_path',
            '../Data/Musite_data/ptm', 'path to data dir')
    flags.DEFINE_string('res_path',
            'res', 'path to result dir')

    # Model parameters.
    flags.DEFINE_bool("multilabel", True, "multilabel or not (default: True)")
    flags.DEFINE_bool("binary", False, "Binary or not (default: False)")
    flags.DEFINE_bool("single_binary", False, "Binary or not (default: False)")
    flags.DEFINE_bool("spec_neg_sam", True, "use ptm-specific negative sampling or not (default: True)")
    flags.DEFINE_bool("class_weights", False, "use class weights or not (default: True)")
    flags.DEFINE_bool("graph", False, "use only partial sequence (default: False)")
    flags.DEFINE_bool("split_head", False, "split head to global and local attention (default: False)")
    flags.DEFINE_bool("no_pdb", False, "not use any pdb (default: False)")

    # Training parameters.
    flags.DEFINE_integer("seq_len", 512, "maximum lenth+2 of the model sequence (default: 512)")
    flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 32)")# 128 for 514; 32 for 1026; 8 for 2050
    flags.DEFINE_integer("num_epochs",
            100, "Number of training epochs (default: 20)")
    flags.DEFINE_integer("n_lstm", 3, "number of lstm layer for rnn model")
    flags.DEFINE_integer("n_gcn", 3, "number of gcn layer")
    flags.DEFINE_integer("fill_cont", 4, "how many sequence should be considered as neighbour/2")
    flags.DEFINE_integer("global_heads", 2, "number of heads allocated to global attention")


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

class PTMDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_name, FLAGS, shuffle=True,ind=None, eval=False, binary=False):
        records = []
        self.ind = ind
        self.eval = eval
        self.shuffle = shuffle
        self.chunk_size = FLAGS.seq_len-2
        self.graph = FLAGS.graph
        self.num_cont = FLAGS.fill_cont
        self.d_model = 128
        self.batch_size = FLAGS.batch_size    
        self.model = FLAGS.model
        self.binary = binary
        # ptm_list = ['Phos_ST','Phos_Y','glyco_N','glyco_ST','Ubi_K','SUMO_K','N6-ace_K','Methy_R','Methy_K','Pyro_Q','Palm_C','Hydro_P','Hydro_K']
        # convert_ptm = {p:i+1 for i, p in enumerate(ptm_list)}
        # flatten = lambda l: [item for sublist in l for item in sublist]
        # Read data and cut into pieces, 
        with open(file_name, 'r') as fp:
            # data structure: {PID:{seq, label:[{site, ptm_type}]}}
            dat = json.load(fp)
            print('loading data')
            for k in tqdm(dat):
                # some case that the data miss sequence, skip
                if dat[k].get('seq',-1)==-1:
                    continue
                self.cut_protein(dat,records, k, chunk_size=FLAGS.seq_len-2)
        # PTM label to corresponding amino acids
        self.label2aa = {'Hydro_K':'K','Hydro_P':'P','Methy_K':'K','Methy_R':'R','N6-ace_K':'K','Palm_C':'C',
        'Phos_ST':'ST','Phos_Y':'Y','Pyro_Q':'Q','SUMO_K':'K','Ubi_K':'K','glyco_N':'N','glyco_ST':'ST'}
        # get unique labels
        self.unique_labels = sorted(set([l['ptm_type'] for d in records for l in d['label'] ]))
        self.label_to_index = {str(label): i for i, label in enumerate(self.unique_labels)}
        self.index_to_label = {i: str(label) for i, label in enumerate(self.unique_labels)}

        self.seq = [d['seq'] for d in records]
        self.label = [ d['label'] for d in records]
        self.uid = [d['uid'] for d in records]

        # save memory
        records = None

        self.list_id = np.arange(len(self.uid))

        # Tokenize the sequence
        self.X = tokenize_seqs(self.seq)
        # construct the raw Y and sample weights
        self.Y = []
        self.sample_weights = []
        print('Constructing Y and sample weights')
        y_keep = []
        for i in tqdm(range(len(self.X))):
            Y = np.zeros((len(self.X[i]), len(self.unique_labels))) if not self.binary \
                else np.zeros((len(self.X[i]), 1))
            sample_weights = np.zeros((len(self.X[i]), len(self.unique_labels))) if not self.binary \
                else np.zeros((len(self.X[i]), 1))
            for lbl in self.label[i]:
                if '~' in self.uid[i]:
                    n_seq = int(self.uid[i].split('~')[1])
                    # minus i * half chunk size
                    site_num = int(lbl['site'])-n_seq*self.chunk_size//2+1
                else:
                    site_num = int(lbl['site'])+1
                if self.binary:
                    if lbl['ptm_type'] == self.unique_labels[self.ind]:
                        Y[site_num, 0] = 1 
                        sample_weights[site_num, 0] = 1
                    else:
                        continue
                else:
                    Y[site_num, self.label_to_index[lbl['ptm_type']]] = 1 
                    sample_weights[site_num, self.label_to_index[lbl['ptm_type']]] = 1

            if self.binary:
                if np.sum(Y)==0:
                    continue
                else:

                    y_keep.append(i) # save index for X and uid
                    pos_y = np.sum(Y, axis = 1).astype(int) >0
                    Y = Y[pos_y]
                    sample_weights = sample_weights[pos_y]
                    self.Y.append(Y)
                    self.sample_weights.append(sample_weights)
            else:
                self.Y.append(Y)
                self.sample_weights.append(sample_weights)                
        if self.binary:
            self.X = [self.X[i] for i in y_keep]
            self.uid = [self.uid[i] for i in y_keep]
            self.seq = [self.seq[i] for i in y_keep]
            self.label = [self.label[i] for i in y_keep]
        # get graph
        if self.graph:
            self.adjs = get_graph(self.uid, self.X, self.num_cont , self.chunk_size, FLAGS.split_head, FLAGS.no_pdb)

        logging.info('Loaded {} records from {}.'.format(len(self.seq),file_name))


        self.on_epoch_end()

    def update_unique_labels(self, unique_labels):
        self.unique_labels = unique_labels
    
    def cut_protein(self, dat,records, k, chunk_size):
        assert chunk_size%4 == 0
        quar_chunk_size = chunk_size//4
        half_chunk_size = chunk_size//2
        sequence = str(dat[k]['seq'])
        labels = dat[k]['label']
        if len(sequence) > chunk_size:
            label_count = 0
            for i in range((len(sequence)-1)//half_chunk_size):
                # the number of half chunks=(len(sequence)-1)//chunk_size+1, minus one because the first chunks contains two halfchunks
                max_seq_ind = (i+2)*half_chunk_size
                if i==0:
                    cover_range = (0,quar_chunk_size*3)
                elif i==((len(sequence)-1)//half_chunk_size-1):
                    cover_range = (quar_chunk_size+i*half_chunk_size, len(sequence))
                    max_seq_ind = len(sequence)
                else:
                    cover_range = (quar_chunk_size+i*half_chunk_size, quar_chunk_size+(i+1)*half_chunk_size)
                sub_labels = [lbl for lbl in labels if (lbl['site'] >= cover_range[0] and lbl['site'] < cover_range[1])]
                if len(sub_labels)==0:
                    continue
                records.append({
                    'uid': k+'~'+str(i),
                    'seq': sequence[i*half_chunk_size: max_seq_ind],
                    'label': sub_labels,
                })
                label_count+=len(sub_labels)
        else:
            records.append({
                'uid': k,
                'seq': sequence,
                'label':labels
            })

    
    def __len__(self):
        # number of batch per epoch
        return math.ceil(len(self.index) / self.batch_size )

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.list_id[k] for k in index]
        X, y, sample_weights = self.__get_data(batch)
        return (X, y, sample_weights)

    def on_epoch_end(self):
        self.index = np.arange(len(self.uid))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        # Tokenize X
        # X = [x[batch] for x in self.X]  # get batch for all component of X
        X = [self.X[b] for b in batch]
        Y = [self.Y[b] for b in batch]
        sample_weights = [self.sample_weights[b] for b in batch]
        uid = [self.uid[b] for b in batch]
        # padding for X, Y, sw
        max_batch_seq_len = max([len(x) for x in X])
        X = self.pad_X(X, max_batch_seq_len)
        Y = self.pad_Y_sw(Y, max_batch_seq_len)
        sample_weights = self.pad_Y_sw(sample_weights, max_batch_seq_len)

        for i in range(len(X)):
            if self.eval:
                # for evaluation, get all negative sample
                if self.binary:
                    # binary
                    u = self.unique_labels[self.ind]
                    aa = self.label2aa[u]
                    seq = [index_to_token[j] for j in X[i]]
                    if '~' in uid[i]:
                        n_seq = int(uid[i].split('~')[1])
                        if n_seq==0:
                            # first chunk includes the first quarter chunk
                            neg_index = np.array([j for j,amino in enumerate(seq[0:self.chunk_size//4*3]) if amino in aa])
                        elif (n_seq+1)*self.chunk_size//2 >len(seq):
                            # last chunk includes the last leftover chunk
                            neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4:len(seq)-1]) if amino in aa])
                        else:
                            # other chunk takes only the middle half chunk to avoid duplication in negative sample
                            neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4: self.chunk_size//4*3]) if amino in aa])
                    else:
                        neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa])
                        if len(neg_index)<1:
                            continue
                        sample_weights[i, neg_index+1, 0] = 1
                else:
                    # multi class
                    for u in self.unique_labels:
                        aa = self.label2aa[u]
                        seq = [index_to_token[j] for j in X[i]]
                        if '~' in uid[i]:
                            n_seq = int(uid[i].split('~')[1])
                            if n_seq==0:
                                # first chunk includes the first quarter chunk
                                neg_index = np.array([j for j,amino in enumerate(seq[0:self.chunk_size//4*3]) if amino in aa])
                            elif (n_seq+1)*self.chunk_size//2 >len(seq):
                                # last chunk includes the last leftover chunk
                                neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4: len(seq)-1]) if amino in aa])
                            else:
                                # other chunk takes only the middle half chunk to avoid duplication in negative sample
                                neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4:self.chunk_size//4*3]) if amino in aa])                            
                        else:
                            neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa])
                        if len(neg_index)<1:
                            continue
                        sample_weights[i, neg_index+1, self.label_to_index[u]] = 1
            else:
                # for training, get randomly selected neg samples.
                if self.binary:
                    # only get the specific PTM negative 
                    u = self.unique_labels[self.ind]
                    aa = self.label2aa[u]
                    seq = [index_to_token[j] for j in X[i]]  
                    if '~' in uid[i]:
                        n_seq = int(uid[i].split('~')[1])
                        if n_seq==0:
                            # first chunk includes the first quarter chunk
                            all_neg_index = np.array([j for j,amino in enumerate(seq[0:self.chunk_size//4*3]) if amino in aa])
                        elif (n_seq+1)*self.chunk_size//2 >len(seq):
                            # last chunk includes the last leftover chunk
                            all_neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4: len(seq)-1]) if amino in aa])
                        else:
                            # other chunk takes only the middle half chunk to avoid duplication in negative sample
                            all_neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4:self.chunk_size//4*3]) if amino in aa])
                    else:
                        all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
                    # remove positive labels
                    pos_index = set(np.array([j for j in range(Y.shape[1]) if Y[i,j, 0]==1], dtype = int))
                    if len(pos_index)==0:
                        continue
                    all_neg_index = set(all_neg_index) - pos_index
                    all_neg_index = np.array([j for j in all_neg_index])
                    # sampling
                    n_pos = int(np.sum(sample_weights[i,:,0]))
                    if len(all_neg_index)<1 or n_pos==0:
                        continue
                    if n_pos <= len(all_neg_index):#if more pos than neg
                        neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
                    else:
                        neg_index = all_neg_index#[0]
                    sample_weights[i, neg_index, 0] = 1
                else:
                    for u in self.unique_labels:
                        aa = self.label2aa[u]
                        seq = [index_to_token[j] for j in X[i]] 
                        if '~' in uid[i]:
                            n_seq = int(uid[i].split('~')[1])
                            if n_seq==0:
                                # first chunk includes the first quarter chunk
                                all_neg_index = np.array([j for j,amino in enumerate(seq[0:self.chunk_size//4*3]) if amino in aa])
                            elif (n_seq+1)*self.chunk_size//2 >len(seq):
                                # last chunk includes the last leftover chunk
                                all_neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4: len(seq)-1]) if amino in aa])
                            else:
                                # other chunk takes only the middle half chunk to avoid duplication in negative sample
                                all_neg_index = np.array([j for j,amino in enumerate(seq[self.chunk_size//4: self.chunk_size//4*3]) if amino in aa])
                        else:
                            all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
                        # remove positive labels
                        pos_index = set(np.array([j for j in range(Y.shape[1]) if Y[i,j, self.label_to_index[u]]==1], dtype = int))
                        all_neg_index = set(all_neg_index) - pos_index
                        all_neg_index = np.array([j for j in all_neg_index])
                        # sampling
                        n_pos = int(np.sum(sample_weights[i,:,self.label_to_index[u]]))
                        if len(all_neg_index)<1 or n_pos==0:
                            continue
                        if n_pos <= len(all_neg_index):#if more pos than neg
                            neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
                        else:
                            neg_index = all_neg_index#[0]
                        sample_weights[i, neg_index, self.label_to_index[u]] = 1
        y = Y.reshape((Y.shape[0],-1,1))
        # if class_weights is None:
        #     class_weights = np.repeat([1], len(unique_labels))
        # if not evaluate:
        #     sample_weights = np.tile(class_weights, (len(data_seq), seq_len, 1)) * sample_weights
        sample_weights = sample_weights.reshape((sample_weights.shape[0], -1,1))

        # if self.binary:
        #     y = y.reshape((y.shape[0], max_batch_seq_len, -1))
        #     y = y[:,:,self.ind]
        #     sample_weights = sample_weights.reshape((y.shape[0],max_batch_seq_len, -1))
        #     sample_weights = sample_weights[:,:,self.ind]
        #     yp = np.sum(y, axis=1)
        #     y = y[yp>0]
        #     sample_weights = sample_weights[yp>0]
        #     X = X[yp>0]
        #     uid = uid[yp>0]
        #     y = np.expand_dims(y,-1)
        #     sample_weights = np.expand_dims(sample_weights,-1)

        X = [X]
        if self.graph:
            adjs = [self.adjs[b] for b in batch]
            adjs = self.pad_adjs(adjs, max_batch_seq_len)
            adjs = np.array(adjs)
            X.append(adjs)
        
        if self.model=='Transformer':
            pos_encoding = positional_encoding(max_batch_seq_len,
                                                self.d_model)
            X.append(tf.tile(pos_encoding, [y.shape[0],1,1]))

        return (X, y, sample_weights)
        # # change shape to (batch, seq_len, labels)
        # y = y.reshape((self.batch_size, max_batch_seq_len, -1))
        # sample_weights = sample_weights.reshape((self.batch_size, max_batch_seq_len, -1))

        # # To randomly assign negative sample everytime
        # for i in range(y.shape[0]):
        #     if self.binary:
        #         assert len(y.shape)==3
        #         assert len(sample_weights.shape)==3
        #         # get all aa
        #         aa = self.label2aa[self.index_to_label[self.ind]]
        #         seq = [index_to_token[j] for j in X[0][i]]  
        #         all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
        #         # remove positive labels
        #         pos_index = set(np.array([j for j in range(y.shape[1]) if y[i,j, 0]==1], dtype = int))
        #         all_neg_index = set(all_neg_index) - pos_index
        #         all_neg_index = np.array([j for j in all_neg_index])
        #         n_pos = int(np.sum(sample_weights[i,:,0]))
        #         if len(all_neg_index)<1 or n_pos==0:
        #             continue
        #         if n_pos <= len(all_neg_index):#if more pos than neg
        #             neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
        #         else:
        #             neg_index = all_neg_index#[0]
        #         sample_weights[i, neg_index, 0] = 1
                
        #         # testing TODO: remove
        #         # if n_pos <= len(all_neg_index):
        #         #     pos_y = np.sum(y[i,:, label_to_index[u]][sample_weights[i, :, label_to_index[u]]==1] ==1)
        #         #     neg_y = np.sum(y[i,:, label_to_index[u]][sample_weights[i, :, label_to_index[u]]==1] ==0)
        #         #     # print(pos_y, neg_y)
        #         #     #assert pos_y == neg_y
        #     else:    
        #         for u in self.unique_labels:
        #             aa = self.label2aa[u]
        #             seq = [index_to_token[j] for j in X[0][i]]  
        #             all_neg_index = np.array([j for j,amino in enumerate(seq) if amino in aa], dtype=int)
        #             # remove positive labels
        #             pos_index = set(np.array([j for j in range(y.shape[1]) if y[i,j, self.label_to_index[u]]==1], dtype = int))
        #             all_neg_index = set(all_neg_index) - pos_index
        #             all_neg_index = np.array([j for j in all_neg_index])
        #             n_pos = int(np.sum(sample_weights[i,:,self.label_to_index[u]]))
        #             if len(all_neg_index)<1 or n_pos==0:
        #                 continue
        #             if n_pos <= len(all_neg_index):#if more pos than neg
        #                 neg_index = np.array(random.sample(set(all_neg_index), k=n_pos), dtype=int)#n_pos TODO
        #             else:
        #                 neg_index = all_neg_index#[0]
        #             sample_weights[i, neg_index, self.label_to_index[u]] = 1
        #             # testing TODO: remove
        #             # if n_pos <= len(all_neg_index):
        #             #     pos_y = np.sum(y[i,:, label_to_index[u]][sample_weights[i, :, label_to_index[u]]==1] ==1)
        #             #     neg_y = np.sum(y[i,:, label_to_index[u]][sample_weights[i, :, label_to_index[u]]==1] ==0)
        #                 # print(pos_y, neg_y)
        #                 #assert pos_y == neg_y

        # y = y.reshape((len(batch),-1, 1))
        # sample_weights = sample_weights.reshape((len(batch), -1, 1))# == 1
        
        # # add graph
        # if self.graph:
        #     adjs = get_graph(uid,X, self.num_cont,self.seq_len)
        #     adjs = np.array(adjs)
        #     X.append(adjs)
        # return (X, y, sample_weights)
    def pad_X(self, X, seq_len):
        return np.array([seq_tokens + (seq_len - len(seq_tokens)) * [additional_token_to_index['<PAD>']] for seq_tokens in X])
    
    def pad_Y_sw(self, Y, seq_len):
        num_label = len(self.unique_labels) if not self.binary else 1
        return np.array([np.concatenate((y, np.zeros((seq_len - len(y), num_label))))  for y in Y])
    
    def pad_adjs(self, adjs, seq_len):
        pad_adjs = []
        for adj in adjs:
            adj = sparse.csr_matrix.todense(adj)
            pad_adj = np.zeros((seq_len, seq_len))
            pad_adj[1:(1+adj.shape[0]),1:(1+adj.shape[0])] = adj
            pad_adjs.append(pad_adj)
        
        return np.array(pad_adjs)

def get_graph(uid,X, num_cont, chunk_size, split_head, no_pdb):
    adjs = []
    adj_dir = '/workspace/PTM/Data/Musite_data/Structure/pdb/AF_cont_map/'
    print('constructing graphs')
    for i in tqdm(range(len(uid))):
        if '~' in uid[i]:
            n_seq = int(uid[i].split('~')[1])
            tuid = uid[i].split('~')[0]
            if exists(adj_dir+tuid+'.cont_map.npy') and not no_pdb:
                adj = np.load(adj_dir+tuid+'.cont_map.npy')
                adj = assign_neighbour(adj, num_cont, split_head)
                n = adj.shape[0]
                left_slice = n_seq*chunk_size//2
                right_slice = min((n_seq+2)*chunk_size//2, n)
                adj = adj[left_slice:right_slice, left_slice:right_slice]
            else:
                n = np.where(np.array(X[i])==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, num_cont, split_head)
        else:
            if exists(adj_dir+uid[i]+'.cont_map.npy') and not no_pdb:
                adj = np.load(adj_dir+uid[i]+'.cont_map.npy')
                adj = assign_neighbour(adj, num_cont, split_head)
                # n = adj.shape[0]
                # # pad adj with [0] as 0 for start 
                # pad_adj[1:(1+n),1:(1+n)] = adj
            else:
                # 24 is the stop sign
                n = np.where(np.array(X[i])==24)[0][0]-1
                adj = np.zeros((n,n))
                adj = assign_neighbour(adj, num_cont, split_head)
        adj = sparse.csr_matrix(adj)
        adjs.append(adj)
    adjs = np.array(adjs)
    return adjs

def assign_neighbour(adj, num_cont, split_head):
    # assign the 2*num_cont neighbours to adj 
    if split_head:
        return adj
    n= adj.shape[0]
    if num_cont>=n:
        return np.ones((n,n), dtype=int)
    if num_cont==0:
        return adj
    for i in range(n):
        left = min(i-num_cont,0)
        right = max(i+num_cont+1, n)
        adj[i,left:right] = 1
    # adj = adj + np.tri(n,k=num_cont) - np.tri(n, k=-(num_cont+1))
    # adj[adj>0] = 1
    return adj


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



def tokenize_seqs(seqs):
    # Note that tokenize_seq already adds <START> and <END> tokens.
    return [seq_tokens for seq_tokens in map(tokenize_seq, seqs)]

def retri_seq(site, half_len, seq, other_token_index):
    if site < half_len:
        left_s = seq[0:site+1]
    else:
        left_s = seq[(site-half_len):(site+1)]
    if site + half_len > len(seq):
        right_s = seq[(site+1):len(seq)]
    else:
        right_s = seq[(site+1):(site+half_len+1)]
    left_s = [additional_token_to_index['<START>']] + [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(left_s)]
    right_s = [aa_to_token_index.get(aa, other_token_index) for aa in parse_seq(right_s)] + [additional_token_to_index['<END>']]
    left_s = (half_len+2 - len(left_s)) * [additional_token_to_index['<PAD>']] + left_s
    right_s = right_s + (half_len+1 - len(right_s)) * [additional_token_to_index['<PAD>']]
    s = left_s + right_s    

    return s

class CategoricalTruePositives(tf.keras.metrics.Metric):

    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.batch_size = batch_size
        self.num_classes = num_classes    

        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight):     

        y_mask = tf.reshape(sample_weight, [-1, 13])
        y_true = tf.reshape(y_true,[-1,  13])
        y_pred = tf.reshape(y_pred,[-1,  13])

        y_trues = [y_true[:,i][y_mask[:,i]==1] for i in range(13)]
        y_preds = [y_pred[:,i][y_mask[:,i]==1] for i in range(13)]

        for i in range(13):
            print(pd.DataFrame(confusion_matrix(y_trues[i], y_preds[i]>=0.5, labels = np.array([0,1])), index = ['0','1'],columns = ['0','1']))

    def result(self):

        return None#self.cat_true_positives